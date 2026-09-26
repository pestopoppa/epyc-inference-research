"""OAB-10 (fixed-overhead trim) and OAB-11 (lane guard) for the opencode actor seat, offline.

What must hold:
* trim: the call never loads the lane's AGENTS.md / CLAUDE.md / CONTEXT.md (the env switch
  opencode 1.18.31 actually reads), the skill catalog and tool go, and the author keeps
  only the file's code-style lines -- never its "autonomous agents: STOP" policy;
* guard: build/compile/benchmark commands are DENIED (explicitly: `--auto` approves every
  ask no rule denies), reads of the anchor SOURCE are denied while its build dirs stay
  readable, the planner and critic cannot write anything, the author can still edit its
  lane and nothing outside it, and nothing re-opens the global bash deny-list;
* prompt/card: under the guard the lane is named as THE source tree and the build dir as
  the anchor BINARY; with every knob off the prompt stays byte-identical to the control.

The permission checks run the config through `_Opencode`, a replica of the binary's own
evaluation (`Wildcard.match`, `Permission.fromConfig/merge/evaluate/disabled`, the
global-then-OPENCODE_CONFIG deep merge), documented in `actor_opencode_config`.
"""
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actor_context, actor_opencode_config as aoc, actors
from autokernel.loop import test_actor_context as fx

#: A slice of the real ~/.config/opencode/opencode.jsonc deny-list (INC-20260823).
GLOBAL = {"permission": {"bash": {"mount*": "deny", "sudo mount*": "deny",
                                  "systemctl*": "deny", "dd of=/dev/*": "deny"}}}
#: opencode's build-agent defaults (the plain seat runs the default `build` agent).
DEFAULTS = {"*": "allow", "doom_loop": "ask",
            "external_directory": {"*": "ask", "/tmp/opencode/*": "allow"},
            "question": "deny", "read": {"*": "allow", "*.env": "ask"}}
BUILD_AGENT = {"question": "allow"}


def _wildcard(text: str, pattern: str) -> bool:
    regex = re.sub(r"[.+^${}()|\[\]\\]", lambda m: "\\" + m.group(0), pattern)
    regex = regex.replace("*", ".*").replace("?", ".")
    if regex.endswith(" .*"):
        regex = regex[:-3] + "( .*)?"
    return re.fullmatch(regex, text, re.S) is not None


def _rules(config: dict) -> list[tuple[str, str, str]]:
    out = []
    for permission, value in config.items():
        for pattern, action in ({"*": value} if isinstance(value, str) else value).items():
            out.append((permission, pattern, action))
    return out


def _merge_deep(a: dict, b: dict) -> dict:
    out = dict(a)
    for key, value in b.items():
        out[key] = (_merge_deep(a[key], value)
                    if isinstance(a.get(key), dict) and isinstance(value, dict) else value)
    return out


class _Opencode:
    """The ruleset one plain-seat call runs under: defaults, then the global config
    deep-merged with OPENCODE_CONFIG, then (for a bounded agent) the agent's own rules."""

    def __init__(self, seat_config: dict | None, agent: dict | None = None):
        merged = _merge_deep(GLOBAL, seat_config or {})
        self.rules = (_rules(DEFAULTS) + _rules(BUILD_AGENT)
                      + _rules(merged.get("permission", {})) + _rules(agent or {}))

    def action(self, permission: str, pattern: str) -> str:
        hit = [r for r in self.rules if _wildcard(permission, r[0]) and _wildcard(pattern, r[1])]
        return hit[-1][2] if hit else "ask"

    def bash(self, *commands: str) -> str:
        """One tool call parses into commands; any denied command denies the call."""
        actions = [self.action("bash", c) for c in commands]
        return "deny" if "deny" in actions else ("ask" if "ask" in actions else "allow")

    def disabled(self, tool: str) -> bool:
        permission = "edit" if tool in ("edit", "write", "apply_patch") else tool
        hit = [r for r in self.rules if _wildcard(permission, r[0])]
        return bool(hit) and hit[-1][1] == "*" and hit[-1][2] == "deny"


def _anchor(tmp: Path) -> tuple[Path, Path]:
    """A fake anchor kernel tree: git root with source dirs and two build dirs."""
    root = tmp / "llama.cpp-experimental-x"
    for sub in (".git", "ggml/src", "src", "gguf-py", "build-cpu/bin", "build-cpu-prof/bin"):
        (root / sub).mkdir(parents=True)
    (root / "AGENTS.md").write_text("upstream policy\n")
    (root / "build-xcframework.sh").write_text("#!/bin/sh\n")   # a FILE named build*
    return root, root / "build-cpu"


class _Tmp(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root, self.build = _anchor(self.tmp)
        self.lane = self.tmp / "targets" / "abc" / "workers" / "lane0"
        self.lane.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def plain(self, role, **knobs):
        return aoc.build_plain_config(role=role, lane=self.lane, build_dir=self.build,
                                      author_note_path=self.tmp / "note.md", **knobs)


ALL_ON = {"trim_instructions": True, "trim_tools": True, "lane_guard": True}


class TrimInstructions(_Tmp):

    def test_the_switch_is_the_env_var_opencode_actually_reads(self):
        self.assertEqual(aoc.TRIM_ENV["OPENCODE_DISABLE_PROJECT_CONFIG"], "1")
        self.assertEqual(aoc.TRIM_ENV["OPENCODE_DISABLE_CLAUDE_CODE"], "1")
        self.assertEqual(aoc.TRIM_ENV["OPENCODE_DISABLE_EXTERNAL_SKILLS"], "1")

    def test_no_instruction_file_of_the_lane_or_anchor_is_ever_listed(self):
        for role in aoc.PLAIN_ROLES:
            cfg = self.plain(role, **ALL_ON) or {}
            listed = cfg.get("instructions", [])
            self.assertFalse([p for p in listed if Path(p).name in
                              ("AGENTS.md", "CLAUDE.md", "CONTEXT.md", "copilot-instructions.md")])
            self.assertFalse([p for p in listed if str(self.lane) in p or str(self.root) in p])

    def test_skill_catalog_and_tool_are_dropped(self):
        for role in aoc.PLAIN_ROLES:
            oc = _Opencode(self.plain(role, trim_instructions=True))
            self.assertTrue(oc.disabled("skill"), role)

    def test_only_the_author_gets_the_style_note_and_it_carries_no_policy(self):
        self.assertNotIn("instructions", self.plain("planner", trim_instructions=True))
        self.assertNotIn("instructions", self.plain("critic", trim_instructions=True))
        cfg = self.plain("author", trim_instructions=True)
        self.assertEqual(cfg["instructions"], [str(self.tmp / "note.md")])
        note = aoc.AUTHOR_STYLE_NOTE
        self.assertIn("ASCII only", note)
        for banned in ("STOP", "autonomous", "PAUSE", "Guide, don't solve", "pull request"):
            self.assertNotIn(banned, note)
        self.assertEqual(note.encode("ascii").decode(), note)

    def test_seat_env_carries_the_switches_and_writes_the_config_beside_the_lane(self):
        seat = actors.ActorSeat(bounded=False, trim_instructions=True)
        env = actors._seat_call(seat, actors.backend_for("q/m", "high"), "author",
                                self.lane, {})
        for key, value in aoc.TRIM_ENV.items():
            self.assertEqual(env[key], value)
        self.assertEqual(env[actors.SEAT_ENV_ARM], "plain+trim-instr")
        config = Path(env["OPENCODE_CONFIG"])
        self.assertEqual(config.parent.parent, self.lane.parent / actors.SEAT_CONFIG_DIR,
                         "never inside the worktree")
        self.assertFalse(list(self.lane.iterdir()))
        body = json.loads(config.read_text())
        self.assertEqual(Path(body["instructions"][0]).read_text().rstrip("\n"),
                         aoc.AUTHOR_STYLE_NOTE)

    def test_every_knob_off_adds_only_the_snapshot_off_config(self):
        """Operator 2026-09-25: the plain seat ALWAYS gets a per-call config, because
        snapshot tracking bloats opencode.db whatever the knobs say."""
        for seat in (None, actors.ActorSeat(bounded=False)):
            for role in aoc.PLAIN_ROLES:
                env = actors._seat_call(seat, actors.backend_for("q/m", "high"),
                                        role, self.lane, {})
                fx.assert_snapshot_only(self, env, fx.config_body(env), self.lane, role=role)
                self.assertEqual(Path(env["OPENCODE_CONFIG"]).name,
                                 f"actor-opencode-plain-{role}.json")
        self.assertFalse(list(self.lane.iterdir()))
        self.assertIsNone(actors._seat_call(actors.ActorSeat(bounded=False, **ALL_ON),
                                            actors.backend_for("gpt-5.6-sol", "high"),
                                            "planner", self.lane, {}),
                          "codex/claude are never touched")


class TrimTools(_Tmp):

    def test_unused_tools_leave_the_request(self):
        oc = _Opencode(self.plain("planner", trim_tools=True))
        for tool in aoc.UNUSED_TOOLS:
            self.assertTrue(oc.disabled(tool), tool)
        for tool in ("bash", "read", "grep", "glob", "edit"):
            self.assertFalse(oc.disabled(tool), tool)

    def test_bounded_fan_out_keeps_task(self):
        perm = aoc.seat_permission("planner", trim_tools=True, keep_task=True)
        self.assertNotIn("task", perm)


class LaneGuardBuilds(_Tmp):

    def test_builds_compiles_and_benchmarks_are_denied_for_every_role(self):
        for role in aoc.PLAIN_ROLES:
            oc = _Opencode(self.plain(role, lane_guard=True))
            for command in ("cmake --build build -j", "make -j 16", "ninja -C build",
                            "gcc -O2 -c x.c -o /tmp/x.o", "g++ -march=native -c y.cpp",
                            "timeout 60 gcc -c x.c", "clang++ -c z.cpp", "cc -c q.c",
                            "/usr/bin/g++ -c y.cpp", "ld -r a b", "ls /tmp/*.o",
                            "llama-bench -m m.gguf", "/x/build/bin/llama-bench -m m",
                            "perf record -g ./a.out", "ctest -R x"):
                self.assertEqual(oc.bash(command), "deny", f"{role}: {command}")
            # `cd build && make`: tree-sitter checks each command separately.
            self.assertEqual(oc.bash("cd build", "make -j"), "deny")

    def test_ordinary_investigation_stays_allowed(self):
        oc = _Opencode(self.plain("planner", lane_guard=True))
        for command in ("grep -rn make_q4_scales ggml/src/ggml-cpu/iqk/",
                        "git log --oneline -5", "git diff --stat", "ls ggml/src",
                        "sed -n '40,120p' ggml/include/gguf.h",
                        "python3 - <<'EOF'\n# make sure the header parses\nprint(1)\nEOF",
                        f"nm -C {self.build}/bin/libggml-cpu.so", "perf report --stdio -i p.data",
                        "grep -c avx512 /proc/cpuinfo", "ls x 2>/dev/null"):
            self.assertEqual(oc.bash(command), "allow", command)

    def test_the_global_deny_list_survives_the_merge(self):
        oc = _Opencode(self.plain("author", **ALL_ON))
        for command in ("sudo mount /dev/sdb1 /mnt/x", "systemctl restart x", "mount -a"):
            self.assertEqual(oc.bash(command), "deny")

    def test_no_bash_rule_is_an_allow(self):
        # The one exception: the author's two GitNexus read forms aimed at the anchor
        # by absolute path (`aoc.gitnexus_allow`), appended after every deny.
        for role in aoc.PLAIN_ROLES:
            perm = self.plain(role, **ALL_ON)["permission"]
            bash_allows = {p for p, a in perm["bash"].items() if a == "allow"}
            self.assertEqual(bash_allows, set(aoc.gitnexus_allow(self.root))
                             if role == "author" else set(), role)
            self.assertEqual(list(perm["bash"])[-len(bash_allows):] if bash_allows else [],
                             list(aoc.gitnexus_allow(self.root)) if bash_allows else [])
            allows = [(k, p) for k, v in perm.items() if isinstance(v, dict)
                      for p, a in v.items() if a == "allow"]
            # The critic's `.env.example` allow only re-opens what its own `.env.*` deny
            # (`CRITIC_READ_RULES`, in place of opencode's built-in ask) closed.
            self.assertTrue(all(k == "external_directory" or (k, p) in
                                {("bash", q) for q in bash_allows}
                                | ({("read", "*.env.example")} if role == "critic" else set())
                                for k, p in allows), allows)


class LaneGuardReads(_Tmp):

    def test_anchor_source_is_fenced_and_its_build_dirs_are_not(self):
        oc = _Opencode(self.plain("planner", lane_guard=True))
        ext = lambda path: oc.action("external_directory", f"{Path(path).parent}/*")  # noqa: E731
        self.assertEqual(ext(self.root / "ggml/src/ggml-cpu/iqk/x.cpp"), "deny")
        self.assertEqual(ext(self.root / "AGENTS.md"), "deny")
        self.assertEqual(ext(self.build / "bin/libggml-cpu.so"), "allow")
        self.assertEqual(ext(self.root / "build-cpu-prof/bin/x"), "allow")
        self.assertEqual(oc.bash(f"cat {self.root}/build-xcframework.sh"), "deny",
                         "a file named build* is source, not a build dir")
        self.assertEqual(ext("/mnt/raid0/llm/tmp/ds41-hypotheses/HYPOTHESES.md"), "ask",
                         "other outside reads keep the default (--auto approves them)")
        for command in (f"grep -rn DequantizerQ4K {self.root}/ggml/src/ggml-cpu/iqk/",
                        f"sed -n 1,80p {self.root}/src/llama.cpp",
                        f"cd {self.root}/gguf-py && python3 x.py", f"grep -rn x {self.root}"):
            self.assertEqual(oc.bash(*command.split(" && ")), "deny", command)

    def test_the_lane_itself_is_never_fenced(self):
        perm = self.plain("planner", lane_guard=True)["permission"]
        text = json.dumps(perm)
        self.assertNotIn(str(self.lane), text)
        inside = self.root / "workers" / "lane0"
        inside.mkdir(parents=True)
        self.assertEqual(aoc.anchor_fence(self.build, inside), (None, (), ()))

    def test_no_build_dir_means_no_read_fence_but_still_no_builds(self):
        perm = aoc.seat_permission("planner", lane=self.lane, lane_guard=True)
        self.assertNotIn("external_directory", perm)
        self.assertEqual(perm["bash"]["make *"], "deny")


class LaneGuardWrites(_Tmp):

    def test_planner_and_critic_are_read_only(self):
        for role in ("planner", "critic"):
            oc = _Opencode(self.plain(role, lane_guard=True))
            self.assertTrue(oc.disabled("edit"), role)
            self.assertTrue(oc.disabled("write"), role)
            for command in ("rm -f x", "cp a /tmp/b", "mkdir /tmp/x", "git checkout -- x",
                            "git stash", "sed -i s/a/b/ f.c", "cat > /tmp/x.c <<EOF\nint x;\nEOF",
                            "echo hi >/tmp/y", "tee /tmp/z"):
                self.assertEqual(oc.bash(command), "deny", f"{role}: {command}")

    def test_the_author_edits_its_lane_and_nothing_else(self):
        oc = _Opencode(self.plain("author", **ALL_ON))
        self.assertFalse(oc.disabled("edit"))
        self.assertFalse(oc.disabled("write"))
        self.assertEqual(oc.action("edit", "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"), "allow")
        self.assertEqual(oc.action("edit", "../../../../tmp/x.c"), "deny")
        self.assertEqual(oc.action("edit", ".git/config"), "deny")
        # The author may still run the read-side git/file commands it needs.
        self.assertEqual(oc.bash("git diff"), "allow")
        self.assertEqual(oc.bash("make -j"), "deny")

    def test_the_author_rule_does_not_reopen_edit_for_native_agents(self):
        perm = self.plain("author", lane_guard=True)["permission"]
        self.assertNotIn("*", perm["edit"])


class BoundedSeatUnderTheKnobs(_Tmp):

    def cfg(self, role, **kw):
        return aoc.build_actor_config(role=role, lane=self.lane, build_dir=self.build,
                                      replace_system_prompt=True, **kw)

    def test_knobs_off_leave_the_bounded_config_unchanged(self):
        self.assertNotIn("permission", self.cfg("planner"))
        self.assertEqual(self.cfg("author")["agent"][aoc.AGENT_NAMES["author"]]
                         ["permission"]["edit"], "allow")

    def test_knobs_on_add_denies_and_a_lane_only_author(self):
        cfg = self.cfg("author", **ALL_ON)
        agent = cfg["agent"][aoc.AGENT_NAMES["author"]]["permission"]
        oc = _Opencode(cfg, agent)
        self.assertEqual(oc.action("edit", "ggml/src/x.c"), "allow")
        self.assertEqual(oc.action("edit", "../x.c"), "deny")
        self.assertEqual(oc.bash("make -j"), "deny")
        self.assertNotIn("task", cfg["permission"], "fan-out owns task")
        self.assertEqual(cfg["permission"]["todowrite"], "deny")


class PromptAndCard(_Tmp):

    def _prompt(self, seat, role="planner", backend=None, context=None):
        seen = {}

        def run(prompt, **kw):
            seen["prompt"], seen["env"] = prompt, kw.get("env")
            seen["config"] = fx.config_body(kw.get("env"))  # call-scoped: read in flight
            return fx.HYPOTHESIS if role == "planner" else '{"paths": ["ggml/src/x.c"]}'

        planner = actors.AgentPlanner(workspace=self.lane, seat=seat,
                                      backend=backend or actors.backend_for("q/m", "high"))
        context = context or {"target": {"recipe": {"backend": "cpu",
                                                    "build_dir": str(self.build)}}}
        with mock.patch.object(actors, "render_context", return_value=fx._real_context_text()), \
                mock.patch.object(actors, "_run_agent", side_effect=run), \
                mock.patch.object(actors.subprocess, "run",
                                  return_value=mock.Mock(stdout=" M ggml/src/x.c\n")):
            if role == "planner":
                planner.propose(context)
            else:
                planner.author(actors.Hypothesis("akm-x", "s", "f", "ggml/src/x.c", "g"), context)
        return seen

    def test_knobs_off_is_the_control_prompt_byte_for_byte(self):
        seen = self._prompt(actors.ActorSeat(bounded=False))
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), fx.CONTROL_SHA256)
        fx.assert_snapshot_only(self, seen["env"], seen["config"], self.lane)

    def test_trim_alone_changes_no_prompt_byte(self):
        seen = self._prompt(actors.ActorSeat(bounded=False, trim_instructions=True,
                                             trim_tools=True))
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), fx.CONTROL_SHA256)

    def test_the_guarded_prompt_names_the_lane_first_and_the_build_as_a_binary(self):
        seen = self._prompt(actors.ActorSeat(bounded=False, **ALL_ON))
        head = seen["prompt"].split("\n\n", 1)[0]
        self.assertTrue(head.startswith("## Source tree and fences"))
        self.assertIn(f"THE source tree is `{self.lane}`", head)
        self.assertIn(f"`{self.build}` is the anchor BINARY build: read-only", head)
        self.assertIn(f"Never read source under `{self.root}`", head)
        self.assertIn("Never build, compile", head)
        self.assertIn("You are read-only", head)
        self.assertTrue(seen["prompt"].endswith(fx._real_prompt()))
        self.assertEqual(seen["env"][actors.SEAT_ENV_ARM],
                         "plain+trim-instr+trim-tools+lane-guard")

    def test_the_guarded_author_is_told_it_edits_only_the_lane(self):
        seen = self._prompt(actors.ActorSeat(bounded=False, lane_guard=True), role="author")
        self.assertIn("Edit only files inside the source tree", seen["prompt"])
        self.assertNotIn("You are read-only", seen["prompt"])

    def test_codex_prompts_are_never_guarded(self):
        seen = self._prompt(actors.ActorSeat(bounded=False, **ALL_ON),
                            backend=actors.backend_for("gpt-5.6-sol", "high"))
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), fx.CONTROL_SHA256)

    def test_the_card_names_the_lane_and_labels_the_build_a_binary(self):
        target = {"recipe": {"backend": "cpu", "build_dir": str(self.build)}}
        plain = actor_context.target_card(target)
        self.assertIn(f"- build dir (anchor binary): {self.build}", plain)
        guarded = actor_context.target_card(target, lane=self.lane)
        self.assertEqual(guarded[0], f"- source tree (THE tree to read and cite; your "
                                     f"working directory): {self.lane}")
        self.assertIn(f"- {actor_context.GUARDED_BUILD_LABEL}: {self.build}", guarded)

    def test_variable_mode_index_carries_the_guarded_card(self):
        seen = self._prompt(actors.ActorSeat(bounded=False, context_mode="variable",
                                             lane_guard=True))
        self.assertIn(f"- source tree (THE tree to read and cite; your working directory): "
                      f"{self.lane}", seen["prompt"])
        self.assertEqual(seen["env"][actors.SEAT_ENV_ARM], "plain+lane-guard+ctx-variable")


class CriticSeat(_Tmp):

    def test_the_guarded_critic_gets_a_read_only_config_and_the_block(self):
        seen = {}

        def run(prompt, **kw):
            seen["prompt"], seen["env"], seen["ro"] = prompt, kw.get("env"), kw.get("read_only")
            seen["config"] = fx.config_body(kw.get("env"))  # call-scoped: read in flight
            return '{"accepted": true}'

        critic = actors.AgentCritic(workspace=self.lane, backend=actors.backend_for("q/m", "high"),
                                    seat=actors.ActorSeat(bounded=False, **ALL_ON))
        context = {"target": {"recipe": {"backend": "cpu", "build_dir": str(self.build)}}}
        with mock.patch.object(actors, "render_context", return_value="ctx"), \
                mock.patch.object(actors, "_run_agent", side_effect=run):
            critic.review_hypothesis(actors.Hypothesis("akm-x", "s", "f", "a.c", "g"), context)
        self.assertTrue(seen["ro"])
        self.assertIn("You are read-only", seen["prompt"])
        body = seen["config"]
        oc = _Opencode(body)
        self.assertTrue(oc.disabled("edit"))
        self.assertEqual(oc.bash("git reset --hard"), "deny")

    def test_an_unseated_critic_carries_only_the_never_ask_block(self):
        seen = {}
        with mock.patch.object(actors, "render_context", return_value="ctx"), \
                mock.patch.object(actors, "_run_agent",
                                  side_effect=lambda p, **kw: seen.update(
                                      env=kw.get("env"), config=fx.config_body(kw.get("env")))
                                  or '{"accepted": true}'):
            actors.AgentCritic(workspace=self.lane, backend=actors.backend_for("q/m", "high")
                               ).review_hypothesis(actors.Hypothesis("a", "s", "f", "a.c", "g"), {})
        fx.assert_snapshot_only(self, seen["env"], seen["config"], self.lane, role="critic")


class Provenance(_Tmp):

    def test_a_plain_config_is_not_recorded_as_a_bounded_seat(self):
        env = {"OPENCODE_CONFIG": str(self.tmp / "c.json"), actors.SEAT_ENV_PLAIN_CONFIG: "1",
               actors.SEAT_ENV_ARM: "plain+lane-guard", **aoc.TRIM_ENV}
        (self.tmp / "c.json").write_text("{}")
        out = actors._seat_provenance(env)
        self.assertEqual(out["seat_config"]["path"], str(self.tmp / "c.json"))
        self.assertEqual(out["seat_env"], aoc.TRIM_ENV)
        self.assertEqual(actors._seat_provenance({}), {})

    def test_a_missing_config_is_evidence_not_a_crash(self):
        out = actors._seat_provenance({"OPENCODE_CONFIG": str(self.tmp / "gone.json")})
        self.assertIn("error", out["seat_config"])


class CliKnobs(unittest.TestCase):

    def test_on_off_map_to_seat_fields(self):
        from autokernel.loop import run
        args = mock.Mock(actor_trim_instructions="on", actor_trim_tools="off",
                         actor_lane_guard="on")
        self.assertEqual(run._actor_knobs(args), {"trim_instructions": True,
                                                  "trim_tools": False, "lane_guard": True})


if __name__ == "__main__":
    unittest.main()
