"""Variable-mode actor context (`actor_context`), tested on the REAL DS41 run-8 prompt.

The fixture `fixtures/ds41-run8-planner-prompt.txt` is the planner prompt opencode
received at 2026-09-24T13:12:03Z (run 8, plain seat), recovered from the opencode
session store (session ses_f2c7495aeffexoV7PMkxf2Txdf). The recovered bytes hashed to
exactly the `prompt.sha256` of that call's `epyc.autokernel.actor_call.v1` record
(`RUN8_RECORDED_SHA256`). ONE edit before commit: the seven 12-13 digit perf
sample-period cells of the hotspot table read `[period redacted]`, because the repo's
PII pre-commit hook classifies such digit runs as account numbers (a false positive,
but the hook is not this lane's to change). Everything else is byte-for-byte. No
inference is spent here.

THE A/B CONTROL PROMPT IS NO LONGER THE RECORDED ONE (deliberate, 2026-09-24).
`render_context` never printed `node_profile`, although run.py puts it in every CPU
context and the CPU directive says "Read node_profile". Run 8's context carried the
retained observation `fixtures/ds41-run8-node-profile.json` (the store file keyed to
anchor ebb68dc55, scope half -- copied verbatim) and the prompt silently dropped it.
`fixtures/ds41-run8-planner-prompt-node-profile.txt` is what run 8 WOULD have sent
with the fix: the recorded prompt plus exactly one inserted section, the rendered
node profile (3,878 chars), between the CPU profile and "## Already tried".
`NodeProfileIsExactlyOneInsertedSection` proves that delta; the recorded fixture
stays as the provenance anchor (its hash joins run 8's call record).
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actor_context, actors

FIXTURE = Path(__file__).with_name("fixtures") / "ds41-run8-planner-prompt.txt"
FIXTURE_SHA256 = "c7e8a321eb993c88779edc9186b075d733ecb370020e28dc6e05ade754f1c262"
NODE_PROFILE_FIXTURE = Path(__file__).with_name("fixtures") / "ds41-run8-node-profile.json"
CONTROL_FIXTURE = Path(__file__).with_name("fixtures") / "ds41-run8-planner-prompt-node-profile.txt"
CONTROL_SHA256 = "a265a03a9f9a24b5feb438055385831a72a183884dfd6450bb945af7d596c993"
NODE_PROFILE_SECTION_CHARS = 3878
RUN8_RECORDED_SHA256 = "72c8677850d955b2598bd9d7254460b0fdead5c0ee2f150c80e897ae0ce47b99"
CPU_FORMAT = dict(
    platform="the CPUs in the selected original serving launch",
    target_path="one source path on the selected CPU serving route",
    profile_rule=("use the original CPU launch/model and inspect its source route; "
                  "if the CPU profile is unavailable, state that limit and do not invent "
                  "timing evidence"))
HYPOTHESIS = json.dumps({"mechanism_id": "akm-x", "statement": "s", "falsifier": "f",
                         "target_surface": "ggml/src/x.c", "target_symbol": "g"})


def _recorded_prompt() -> str:
    """Byte-for-byte what run 8 sent (less seven redactions)."""
    return FIXTURE.read_text(encoding="utf-8")


def _real_prompt() -> str:
    """The A/B control: run 8's prompt as the node_profile-rendering code builds it."""
    return CONTROL_FIXTURE.read_text(encoding="utf-8")


def _run8_node_profile() -> dict:
    return json.loads(NODE_PROFILE_FIXTURE.read_text(encoding="utf-8"))["section"]


def _node_profile_block() -> str:
    """The section text `render_context` inserts: its lines after the leading blank
    (which joins onto the CPU profile's own trailing blank line), plus the blank line
    that separates it from the next header."""
    return "\n".join(actors._render_node_profile(_run8_node_profile())[1:]) + "\n\n"


def _real_context_text(prompt: str | None = None) -> str:
    """The rendered bundle inside the real prompt: everything the template wraps."""
    marker = "\x00CONTEXT\x00"
    prefix, suffix = actors._HYPOTHESIS_TASK.format(context=marker, **CPU_FORMAT).split(marker)
    prompt = _real_prompt() if prompt is None else prompt
    assert prompt.startswith(prefix) and prompt.endswith(suffix)
    return prompt[len(prefix):len(prompt) - len(suffix)]


def _cpu_context() -> dict:
    return {"target": {"recipe": {"backend": "cpu"}}}


class TheFixtureIsTheRealPrompt(unittest.TestCase):

    def test_fixture_bytes_are_the_recorded_run8_prompt_less_seven_redactions(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA256)
        self.assertEqual(_recorded_prompt().count("[period redacted]"), 7)

    def test_node_profile_fixture_is_the_run8_retained_observation(self):
        body = json.loads(NODE_PROFILE_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(body["cache_key"]["anchor_commit"],
                         "ebb68dc55d5f6af4a4a5dccdd2a013fa76c63bee")
        self.assertEqual(body["cache_key"]["scope"], "half",
                         "run 8 is common-scope 'half' work: the half-scope retention")
        self.assertEqual(body["section"]["status"], "observed")


class NodeProfileIsExactlyOneInsertedSection(unittest.TestCase):
    """The ONE deliberate change to the inline prompt: the node profile is rendered."""

    def test_control_fixture_is_the_recorded_prompt_plus_exactly_the_node_profile(self):
        recorded, control, block = _recorded_prompt(), _real_prompt(), _node_profile_block()
        self.assertEqual(hashlib.sha256(control.encode()).hexdigest(), CONTROL_SHA256)
        self.assertEqual(len(block), NODE_PROFILE_SECTION_CHARS)
        self.assertEqual(len(control) - len(recorded), NODE_PROFILE_SECTION_CHARS)
        at = recorded.index("\n\n## Already tried\n") + 2
        self.assertEqual(control, recorded[:at] + block + recorded[at:])
        self.assertTrue(block.startswith(actors.NODE_PROFILE_HEADER + "\n"))

    def test_render_context_inserts_exactly_that_section_and_nothing_else(self):
        base = {"target": {"recipe": {"backend": "cpu"}},
                "program": "directive",
                "cpu_profile": {"status": "unavailable", "reason": "none"},
                "prior_experiments": []}
        without = actors.render_context(base)
        with_np = actors.render_context({**base, "node_profile": _run8_node_profile()})
        block = "\n".join(actors._render_node_profile(_run8_node_profile())) + "\n"
        self.assertEqual(with_np.replace(block, "", 1), without)
        self.assertEqual(with_np.count(actors.NODE_PROFILE_HEADER), 1)
        self.assertLess(with_np.index("## CPU profile"), with_np.index(actors.NODE_PROFILE_HEADER))
        self.assertLess(with_np.index(actors.NODE_PROFILE_HEADER), with_np.index("## Already tried"))

    def test_shares_are_rendered_and_absolute_microseconds_are_not(self):
        text = "\n".join(actors._render_node_profile(_run8_node_profile()))
        for share in ("| 1 | 43.51% | `dense-matmul` | MUL_MAT |",
                      "| 2 | 42.05% | `moe-expert-matmul` | MUL_MAT_ID |",
                      "| 42.05% | `expert_mul_mat_id` |", "| 99.20% | `ctx.graph_compute` |"):
            self.assertIn(share, text)
        for absolute in ("18059", "17454", "104771", "41503"):
            self.assertNotIn(absolute, text, "absolutes do not transfer from the sibling")

    def test_an_absent_profile_prints_its_reason_never_a_zero(self):
        text = actors.render_context({"target": {"recipe": {"backend": "cpu"}},
                                      "node_profile": {"status": "not_collected",
                                                       "reason": "sibling build not completed"}})
        self.assertIn("Node profile not_collected: sibling build not completed", text)
        self.assertNotIn("| rank | wall share", text)

    def test_unmeasured_fault_counts_are_not_printed_as_zero(self):
        observation = {**_run8_node_profile()}
        observation["engram_fault_mix"] = {**observation["engram_fault_mix"],
                                           "fault_counts_are_measured": False}
        text = "\n".join(actors._render_node_profile(observation))
        self.assertIn("UNMEASURED at this level", text)
        self.assertNotIn("(minor / major): 0 / 0", text)

    def test_a_gpu_context_never_renders_it(self):
        text = actors.render_context({"node_profile": _run8_node_profile(), "kernel_hotspots": []})
        self.assertNotIn(actors.NODE_PROFILE_HEADER, text)


#: What a knobs-off plain opencode call now carries (operator 2026-09-25): a per-call
#: config that only turns snapshot tracking off, marked as the plain seat's, and
#: nothing else -- no arm label, no trim switches, and (asserted by the callers) not
#: one changed prompt byte.
SNAPSHOT_ONLY_CONFIG = {"$schema": "https://opencode.ai/config.json", "snapshot": False}


def config_body(env):
    """The per-call OPENCODE_CONFIG's JSON, read while the call is in flight."""
    path = (env or {}).get("OPENCODE_CONFIG")
    return json.loads(Path(path).read_text()) if path else None


def assert_snapshot_only(case, env, body, ws=None):
    case.assertEqual(set(env or {}), {"OPENCODE_CONFIG", actors.SEAT_ENV_PLAIN_CONFIG},
                     "knobs off: the snapshot-off config and nothing else (no arm, no trim)")
    case.assertEqual(env[actors.SEAT_ENV_PLAIN_CONFIG], "1")
    case.assertEqual(body, SNAPSHOT_ONLY_CONFIG)
    if ws is not None:
        case.assertEqual(Path(env["OPENCODE_CONFIG"]).parent, Path(ws).parent,
                         "beside the lane, never inside the worktree")


class InlineModeIsByteIdentical(unittest.TestCase):
    """Inline is the A/B control: it must be exactly what run 8 sent."""

    def _captured(self, seat, backend=None, context_text=None):
        seen = {}
        context_text = _real_context_text() if context_text is None else context_text

        def run(prompt, **kw):
            seen["prompt"], seen["env"] = prompt, kw.get("env")
            seen["config"] = config_body(kw.get("env"))
            return HYPOTHESIS

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"
            ws.mkdir(parents=True)
            planner = actors.AgentPlanner(
                workspace=ws, backend=backend or actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
                seat=seat)
            with mock.patch.object(actors, "render_context", return_value=context_text), \
                    mock.patch.object(actors, "_run_agent", side_effect=run):
                planner.propose(_cpu_context())
            seen["bundle_dirs"] = list((ws.parent / actor_context.BUNDLE_DIR).glob("*"))
        return seen

    def test_the_default_seat_reproduces_the_control_prompt_byte_for_byte(self):
        for seat in (None, actors.ActorSeat(bounded=False), actors.ActorSeat(bounded=False,
                                                                             context_mode="inline")):
            seen = self._captured(seat)
            self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), CONTROL_SHA256)
            assert_snapshot_only(self, seen["env"], seen["config"])
            self.assertEqual(seen["bundle_dirs"], [])

    def test_the_template_still_wraps_the_recorded_run8_bundle_byte_for_byte(self):
        """Only `render_context` changed: the recorded bundle still yields run 8's bytes."""
        seen = self._captured(None, context_text=_real_context_text(_recorded_prompt()))
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), FIXTURE_SHA256)

    def test_variable_mode_is_inline_for_non_opencode_backends(self):
        seen = self._captured(actors.ActorSeat(context_mode="variable"),
                              backend=actors.backend_for("gpt-5.6-sol", "high"))
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), CONTROL_SHA256)
        self.assertEqual(seen["bundle_dirs"], [])


class SectionSplit(unittest.TestCase):

    def test_sections_partition_the_real_bundle_exactly(self):
        recorded = _real_context_text(_recorded_prompt())
        self.assertEqual([s.key for s in actor_context.split_sections(recorded)],
                         ["target", "program", "program_strategy", "profile", "already_tried",
                          "shared_history", "serving_observations", "inbox"])
        text = _real_context_text()
        sections = actor_context.split_sections(text)
        self.assertEqual("".join(s.text for s in sections), text)
        self.assertEqual([s.key for s in sections],
                         ["target", "program", "program_strategy", "profile", "node_profile",
                          "already_tried", "shared_history", "serving_observations", "inbox"])
        self.assertEqual(next(s for s in sections if s.key == "node_profile").text,
                         _node_profile_block())
        program = next(s for s in sections if s.key == "program")
        self.assertIn("CPU COMMON-SCOPE SOURCE WORK: half", program.text)
        self.assertNotIn("# AutoKernel loop", program.text,
                         "program.md itself goes to a file; only the run.py directives stay")

    def test_every_render_context_header_is_recognised(self):
        """Guards drift between `render_context`'s headers and the splitter."""
        prior = [{"mechanism_id": f"akm-barrier-{i}", "status": "measured_null",
                  "target_surface": "ggml/src/a.c", "target_symbol": "f",
                  "effect_fraction": 0.01, "comparable_measurement": True,
                  "mechanism_claim_status": "verified"} for i in range(3)]
        prior.append({"mechanism_id": "akm-old", "status": "superseded", "statement": "s"})
        context = {
            "target": {"recipe": {"backend": "cpu"}},
            "program": "directive\n\n# AutoKernel loop — strategy\n\n## Settled\n- x",
            "cpu_profile": {"status": "unavailable", "reason": "none"},
            "node_profile": _run8_node_profile(),
            "prior_experiments": prior,
            "shared_prior_experiments": {"rows": [{"mechanism_id": "akm-s"}]},
            "serving_observations": {"rows": []},
            "prior_hypothesis_rejections": ["r1"], "prior_patch_rejections": ["r2"],
            "inbox": ["# note\n## Already tried\n- operator prose that repeats a header"],
        }
        text = actors.render_context(context)
        keys = [s.key for s in actor_context.split_sections(text)]
        for key in ("target", "program", "program_strategy", "superseded", "profile",
                    "node_profile", "exhausted_families", "stagnant_families", "already_tried",
                    "shared_history", "serving_observations", "hypothesis_rejections",
                    "patch_rejections", "inbox"):
            self.assertIn(key, keys)
        self.assertEqual(keys.count("already_tried"), 1,
                         "a header inside the inbox must not open a new section")
        self.assertEqual("".join(s.text for s in actor_context.split_sections(text)), text)

    def test_gpu_profile_header_is_the_profile_section(self):
        text = actors.render_context({"kernel_hotspots": [
            {"share_of_device_time": 0.5, "total_duration_ns": 1, "calls": 1, "signature": "k"}]})
        self.assertIn("profile", [s.key for s in actor_context.split_sections(text)])


class VariableModeIsLossless(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.text = _real_context_text()
        self.bundle = actor_context.materialize(self.text, Path(self._tmp.name), role="planner")

    def tearDown(self):
        self._tmp.cleanup()

    def test_section_files_concatenate_to_the_inline_bundle(self):
        files = sorted((self.bundle.directory / "sections").iterdir())
        self.assertEqual("".join(f.read_text(encoding="utf-8") for f in files), self.text)

    def test_json_tree_round_trips_to_the_inline_json(self):
        for section in self.bundle.sections:
            if section.key not in actor_context.JSON_SECTIONS:
                continue
            inline = actor_context.json_payload(section)
            restored = actor_context.implode(self.bundle.directory / "json" / section.key)
            self.assertEqual(restored, inline, section.key)
            self.assertIn(json.dumps(restored, indent=2, sort_keys=True), self.text,
                          f"{section.key}: the restored JSON is the exact text inline shows")

    def test_explode_is_lossless_for_awkward_keys_and_values(self):
        value = {"a/b": [1, 2.5, None, True, "x" * 7000], "A/B": {"é": "ü" * 7000},
                 "_index": {"n": list(range(3000))}, "": {}, "empty": []}
        root = Path(self._tmp.name) / "awkward"
        actor_context.explode(value, root)
        self.assertEqual(actor_context.implode(root), value)

    def test_index_lists_every_section_and_top_level_key_with_sizes(self):
        index = self.bundle.index
        for n, section in enumerate(self.bundle.sections, 1):
            self.assertIn(f"| {n} | {section.key} | `sections/{n:02d}-{section.key}.md`", index)
            self.assertIn(f"{len(section.text):,}", index)
        for section in self.bundle.sections:
            payload = (actor_context.json_payload(section)
                       if section.key in actor_context.JSON_SECTIONS else None)
            for key, value in (payload or {}).items():
                size = len(json.dumps(value, indent=2, sort_keys=True))
                self.assertRegex(index, rf"{key}/? {actor_context._k(size)}\b",
                                 f"{section.key}.{key}")

    def test_inline_sections_are_verbatim_and_file_sections_are_absent(self):
        index = self.bundle.index
        for section in self.bundle.sections:
            if section.inline:
                self.assertIn(section.text.rstrip("\n"), index, section.key)
        self.assertNotIn('"full_transfer_target"', index, "the target JSON stays on disk")
        self.assertNotIn("## Measured gfx90a facts", index.split("=== INLINE")[1])
        self.assertIn("mul_mat_qX_K_q8_2_X4_T", index, "the hotspot table stays inline")
        self.assertIn(str(self.bundle.directory / "INDEX.md"), index)

    def test_node_profile_is_a_required_file_with_its_share_table_inline(self):
        index = self.bundle.index
        n = [s.key for s in self.bundle.sections].index("node_profile") + 1
        rel = f"sections/{n:02d}-node_profile.md"
        self.assertEqual((self.bundle.directory / rel).read_text(encoding="utf-8"),
                         _node_profile_block())
        self.assertRegex(index, rf"\| {n} \| node_profile \| `{rel}` \| 3,878 \| \d+ \| "
                                r"file \+ summary \|")
        required = index.split("Read these before you propose")[1].split("| # |")[0]
        self.assertIn(f"`{rel}` — node_profile", required)
        inline = index.split("=== INLINE sections (verbatim) ===")[1]
        self.assertIn(actors.NODE_PROFILE_HEADER, inline)
        self.assertIn("| 1 | 43.51% | `dense-matmul` | MUL_MAT |", inline)
        self.assertNotIn("### Ops by wall share", inline, "the per-op table is on disk")
        self.assertNotIn("Limitations:", inline.split(actors.NODE_PROFILE_HEADER)[1]
                         .split("## Already tried")[0])
        self.assertIn(f"is `{self.bundle.directory}/{rel}`)", inline)
        head = actor_context.section_summary(
            next(s for s in self.bundle.sections if s.key == "node_profile"))
        self.assertLess(len(head), 1500)

    def test_an_absent_node_profile_is_its_own_summary(self):
        section = actor_context.Section("node_profile", actors.NODE_PROFILE_HEADER
                                        + "\ncaveat\nNode profile absent: no sibling\n\n")
        self.assertEqual(actor_context.section_summary(section), section.text)

    def test_target_card_resolves_dedupe_pointers(self):
        card = "\n".join(actor_context.target_card(actor_context.json_payload(
            self.bundle.sections[0])))
        self.assertIn("model: /mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/"
                      "DeepSeek-V4.1-Flash-Q4.gguf", card)
        self.assertIn("build dir (anchor binary): /mnt/raid0/llm/llama.cpp-experimental-"
                      "deepseek41-20260923/build-cpu", card)
        self.assertIn("threads: 48", card)
        self.assertNotIn("<same as", card)

    def test_the_index_is_a_fraction_of_the_inline_bundle(self):
        self.assertLess(len(self.bundle.index), 0.25 * len(self.text))

    def test_manifest_binds_the_prompt(self):
        prompt = "HEAD\n" + self.bundle.index + "\nTAIL"
        self.bundle.seal(prompt)
        manifest = json.loads((self.bundle.directory / "manifest.json").read_text())
        self.assertEqual(manifest["prompt"]["sha256"], hashlib.sha256(prompt.encode()).hexdigest())
        self.assertEqual(manifest["inline_equivalent_prompt_chars"], len(self.text) + 10)
        for rel, ref in manifest["files"].items():
            data = (self.bundle.directory / rel).read_bytes()
            self.assertEqual(ref["sha256"], hashlib.sha256(data).hexdigest(), rel)


class VariableModeThroughThePlanner(unittest.TestCase):

    def _run(self, seat, role="planner"):
        seen = {}

        def run(prompt, **kw):
            seen["prompt"], seen["env"] = prompt, kw.get("env")
            seen["config"] = config_body(kw.get("env"))
            return HYPOTHESIS if role == "planner" else '{"paths": ["ggml/src/x.c"]}'

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ws = Path(tmp.name) / "workers" / "lane0"
        ws.mkdir(parents=True)
        planner = actors.AgentPlanner(workspace=ws, backend=actors.backend_for("q/m", "high"),
                                      seat=seat)
        with mock.patch.object(actors, "render_context", return_value=_real_context_text()), \
                mock.patch.object(actors, "_run_agent", side_effect=run), \
                mock.patch.object(actors.subprocess, "run",
                                  return_value=mock.Mock(stdout=" M ggml/src/x.c\n")):
            if role == "planner":
                planner.propose(_cpu_context())
            else:
                planner.author(actors.Hypothesis("akm-x", "s", "f", "ggml/src/x.c", "g"),
                               _cpu_context())
        seen["ws"] = ws
        seen["bundles"] = sorted((ws.parent / actor_context.BUNDLE_DIR).glob("*"))
        return seen

    def test_plain_variable_sends_the_index_and_labels_the_arm(self):
        seen = self._run(actors.ActorSeat(bounded=False, context_mode="variable"))
        self.assertEqual(len(seen["bundles"]), 1)
        bundle = seen["bundles"][0]
        self.assertEqual(bundle.parent.parent, seen["ws"].parent, "never inside the worktree")
        self.assertFalse(list(seen["ws"].iterdir()))
        self.assertIn(str(bundle), seen["prompt"])
        self.assertNotIn('"full_transfer_target"', seen["prompt"])
        self.assertTrue(seen["prompt"].endswith('{"abstain": "<specific reason>"}.'))
        self.assertEqual(seen["env"][actors.SEAT_ENV_ARM], "plain+ctx-variable")
        assert_snapshot_only(self, {k: v for k, v in seen["env"].items()
                                    if k != actors.SEAT_ENV_ARM}, seen["config"], seen["ws"])
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(manifest["prompt"]["sha256"],
                         hashlib.sha256(seen["prompt"].encode()).hexdigest())
        self.assertEqual(manifest["inline_equivalent_prompt_chars"], len(_real_prompt()))

    def test_bounded_variable_keeps_the_seat_config_and_suffixes_its_arm(self):
        seen = self._run(actors.ActorSeat(bounded=True, context_mode="variable",
                                          tools_python="/py"))
        self.assertEqual(seen["env"][actors.SEAT_ENV_ARM], "bounded+ctx-variable")
        self.assertIn("OPENCODE_CONFIG", seen["env"])

    def test_author_gets_the_index_too(self):
        seen = self._run(actors.ActorSeat(bounded=False, context_mode="variable"), role="author")
        self.assertIn("-author-", seen["bundles"][0].name)
        self.assertNotIn('"full_transfer_target"', seen["prompt"])
        self.assertIn("DO NOT BUILD", seen["prompt"])

    def test_an_unwritable_bundle_degrades_to_an_honestly_labelled_inline_call(self):
        with mock.patch.object(actor_context, "materialize", side_effect=OSError("disk full")):
            seen = self._run(actors.ActorSeat(bounded=False, context_mode="variable"))
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), CONTROL_SHA256)
        self.assertNotIn(actors.SEAT_ENV_ARM, seen["env"],
                         "an inline call is never recorded as the variable arm")
        assert_snapshot_only(self, seen["env"], seen["config"], seen["ws"])



class OrchestratorVariableMode(unittest.TestCase):
    """INF-78 OAB-7: the bundle rides `ChatRequest.context_bundle` to the orchestrator,
    whose REPL holds it as the variable `context`; the prompt carries the index."""

    #: The orchestrator's payload contract (epyc-orchestrator
    #: src/repl_environment/context_bundle.py `ContextBundle.from_payload`), restated so a
    #: drift fails HERE, in the repo that builds the payload.
    NAME = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
    FENCE = __import__("re").compile(r"```json\n(.*)\n```", __import__("re").S)

    def tearDown(self):
        from autokernel.loop import actor_orchestrator
        actor_orchestrator._PENDING.clear()
        actor_orchestrator._STAGED.clear()

    def test_payload_partitions_the_real_bundle_losslessly(self):
        text = _real_context_text()
        bundle = actor_context.orchestrator_bundle(text, role="planner")
        entries = bundle.payload["sections"]
        self.assertEqual(bundle.payload["schema"], actor_context.ORCH_BUNDLE_SCHEMA)
        self.assertEqual("".join(e["text"] for e in entries), text)
        self.assertEqual([e["name"] for e in entries],
                         [s.key for s in actor_context.split_sections(text)])
        kinds = {e["name"]: e["kind"] for e in entries}
        self.assertEqual({k for k, v in kinds.items() if v == "json"},
                         {"target", "shared_history", "serving_observations"})
        for e in entries:
            self.assertRegex(e["name"], self.NAME)
            self.assertLessEqual(set(e), {"name", "text", "kind", "inline", "description"})
            self.assertLessEqual(len(e["description"]), 300)
            self.assertEqual(e["inline"], e["name"] in actor_context.INLINE_SECTIONS)
            if e["kind"] == "json":   # the orchestrator's parse rule gives the same object
                parsed = json.loads(self.FENCE.search(e["text"]).group(1))
                self.assertEqual(parsed, actor_context.json_payload(actor_context.Section(e["name"], e["text"])))

    def test_index_is_the_inline_set_and_names_no_file(self):
        text = _real_context_text()
        bundle = actor_context.orchestrator_bundle(text, role="planner")
        for section in actor_context.split_sections(text):
            if section.inline:
                self.assertIn(section.text.rstrip("\n"), bundle.index)
        self.assertNotIn('"full_transfer_target"', bundle.index)
        self.assertNotIn(actor_context.BUNDLE_DIR, bundle.index)
        self.assertNotIn("sections/", bundle.index)
        self.assertIn('context.get("node_profile")', bundle.index)
        self.assertIn("`inbox` -- operator suggestions", bundle.index)
        self.assertLess(len(bundle.index), len(text) // 3)

    def _propose(self, backend, seat):
        seen = {}

        def run(prompt, **kw):
            seen["prompt"], seen["env"] = prompt, kw.get("env")
            seen["argv"] = kw["backend"].argv(prompt, kw["workspace"], schema=kw.get("schema"))
            return HYPOTHESIS

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"
            ws.mkdir(parents=True)
            planner = actors.AgentPlanner(workspace=ws, backend=backend, seat=seat)
            with mock.patch.object(actors, "render_context", return_value=_real_context_text()), \
                    mock.patch.object(actors, "_run_agent", side_effect=run):
                planner.propose(_cpu_context())
            argv = seen["argv"]
            if "--context-bundle" in argv:
                seen["bundle"] = json.loads(Path(argv[argv.index("--context-bundle") + 1]).read_text())
            seen["bundle_dirs"] = list((ws.parent / actor_context.BUNDLE_DIR).glob("*"))
        return seen

    def test_the_orchestrator_planner_ships_the_bundle_and_sends_the_index(self):
        seen = self._propose(actors.backend_for("orch:auto", "high"),
                             actors.ActorSeat(bounded=False, context_mode="orchestrator-variable"))
        self.assertIn("ORCHESTRATOR mode", seen["prompt"])
        self.assertNotIn('"full_transfer_target"', seen["prompt"])
        self.assertTrue(seen["prompt"].endswith('{"abstain": "<specific reason>"}.'))
        self.assertEqual(seen["env"][actors.SEAT_ENV_ARM], "orch+ctx-orch-variable")
        self.assertEqual("".join(e["text"] for e in seen["bundle"]["sections"]), _real_context_text())
        manifest = seen["bundle"]["manifest"]
        self.assertEqual(manifest["prompt"]["sha256"], hashlib.sha256(seen["prompt"].encode()).hexdigest())
        self.assertEqual(manifest["inline_equivalent_prompt_chars"], len(_real_prompt()))
        self.assertEqual(seen["bundle_dirs"], [], "no on-disk bundle tree: the REPL holds it")

    def test_inline_stays_the_default_for_the_orchestrator_kind(self):
        for seat in (None, actors.ActorSeat(bounded=False)):
            seen = self._propose(actors.backend_for("orch:auto", "high"), seat)
            self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), CONTROL_SHA256)
            self.assertNotIn("--context-bundle", seen["argv"])

    def test_orchestrator_variable_is_inline_for_every_other_kind(self):
        backend = actors.backend_for("gpt-5.6-sol", "high")
        seen = {}

        def run(prompt, **kw):
            seen["prompt"] = prompt
            return HYPOTHESIS

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"
            ws.mkdir(parents=True)
            planner = actors.AgentPlanner(workspace=ws, backend=backend,
                                          seat=actors.ActorSeat(context_mode="orchestrator-variable"))
            with mock.patch.object(actors, "render_context", return_value=_real_context_text()), \
                    mock.patch.object(actors, "_run_agent", side_effect=run):
                planner.propose(_cpu_context())
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), CONTROL_SHA256)

    def test_argv_attaches_the_bundle_only_for_its_own_prompt(self):
        from autokernel.loop import actor_orchestrator as orch
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"
            ws.mkdir(parents=True)
            backend = actors.backend_for("orch:auto", "high")
            orch.stage_bundle(ws, "THE PROMPT", {"sections": [{"name": "a", "text": "x"}]})
            self.assertNotIn("--context-bundle", backend.argv("A CRITIC PROMPT", ws))
            again = backend.argv("THE PROMPT", ws)              # a retry of the same call
            self.assertIn("--context-bundle", again)
            capped = __import__("dataclasses").replace(backend, context_print_cap_bytes=2048,
                                                       context_pull_budget_bytes=90000)
            argv = capped.argv("THE PROMPT", ws)
            self.assertEqual(argv[argv.index("--context-print-cap-bytes") + 1], "2048")
            self.assertEqual(argv[argv.index("--context-pull-budget-bytes") + 1], "90000")

    def test_collect_projects_the_servers_pull_accounting(self):
        from autokernel.loop import actor_orchestrator as orch
        pulls = {"schema": "epyc.orchestrator.context_pulls.v1",
                 "bundle": {"sha256": "ab" * 32, "sections": 2, "bytes": 900},
                 "print_cap_bytes": 4096, "pull_budget_bytes": None,
                 "offered": {"bytes": 900, "in_prompt_bytes": 100, "variable_only_bytes": 800},
                 "totals": {"pull_calls": 3, "bytes_pulled": 450, "unique_bytes": 400},
                 "sections": {"inbox": {"kind": "text", "in_prompt": False, "offered_bytes": 800,
                                        "pulls": 3, "bytes_pulled": 450, "unique_bytes": 400,
                                        "coverage": 0.5}},
                 "turns": [{"turn": 1, "pulls": [{"op": "get", "section": "inbox", "bytes": 450}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"
            ws.mkdir(parents=True)
            actors.backend_for("orch:auto", "high").argv("P", ws)
            sidecar = orch._PENDING[orch._key(ws)]
            sidecar.write_text(json.dumps({"request": {"schema_sha256": None},
                                           "response": {"turns": 4, "context_pulls": pulls},
                                           "context_bundle_acknowledged": True,
                                           "http_status": 200}))
            stats = orch.collect(ws)
        self.assertEqual(stats["totals"]["bundle_tool_calls"], 3)
        self.assertTrue(stats["context_bundle_acknowledged"])
        self.assertEqual(stats["context_pulls"]["sections"]["inbox"]["coverage"], 0.5)
        self.assertNotIn("turns", stats["context_pulls"], "per-turn records stay in the sidecar")
        self.assertNotIn("totals.bundle_tool_calls", stats["unexposed"])


if __name__ == "__main__":
    unittest.main()
