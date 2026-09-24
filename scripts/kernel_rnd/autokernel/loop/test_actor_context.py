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
RUN8_RECORDED_SHA256 = "72c8677850d955b2598bd9d7254460b0fdead5c0ee2f150c80e897ae0ce47b99"
CPU_FORMAT = dict(
    platform="the CPUs in the selected original serving launch",
    target_path="one source path on the selected CPU serving route",
    profile_rule=("use the original CPU launch/model and inspect its source route; "
                  "if the CPU profile is unavailable, state that limit and do not invent "
                  "timing evidence"))
HYPOTHESIS = json.dumps({"mechanism_id": "akm-x", "statement": "s", "falsifier": "f",
                         "target_surface": "ggml/src/x.c", "target_symbol": "g"})


def _real_prompt() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _real_context_text() -> str:
    """The rendered bundle inside the real prompt: everything the template wraps."""
    marker = "\x00CONTEXT\x00"
    prefix, suffix = actors._HYPOTHESIS_TASK.format(context=marker, **CPU_FORMAT).split(marker)
    prompt = _real_prompt()
    assert prompt.startswith(prefix) and prompt.endswith(suffix)
    return prompt[len(prefix):len(prompt) - len(suffix)]


def _cpu_context() -> dict:
    return {"target": {"recipe": {"backend": "cpu"}}}


class TheFixtureIsTheRealPrompt(unittest.TestCase):

    def test_fixture_bytes_are_the_recorded_run8_prompt_less_seven_redactions(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA256)
        self.assertEqual(_real_prompt().count("[period redacted]"), 7)


class InlineModeIsByteIdentical(unittest.TestCase):
    """Inline is the A/B control: it must be exactly what run 8 sent."""

    def _captured(self, seat, backend=None):
        seen = {}

        def run(prompt, **kw):
            seen["prompt"], seen["env"] = prompt, kw.get("env")
            return HYPOTHESIS

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"
            ws.mkdir(parents=True)
            planner = actors.AgentPlanner(
                workspace=ws, backend=backend or actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
                seat=seat)
            with mock.patch.object(actors, "render_context", return_value=_real_context_text()), \
                    mock.patch.object(actors, "_run_agent", side_effect=run):
                planner.propose(_cpu_context())
            seen["bundle_dirs"] = list((ws.parent / actor_context.BUNDLE_DIR).glob("*"))
        return seen

    def test_the_default_seat_reproduces_the_run8_prompt_byte_for_byte(self):
        for seat in (None, actors.ActorSeat(bounded=False), actors.ActorSeat(bounded=False,
                                                                             context_mode="inline")):
            seen = self._captured(seat)
            self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), FIXTURE_SHA256)
            self.assertIsNone(seen["env"], "plain inline adds no env and no arm suffix")
            self.assertEqual(seen["bundle_dirs"], [])

    def test_variable_mode_is_inline_for_non_opencode_backends(self):
        seen = self._captured(actors.ActorSeat(context_mode="variable"),
                              backend=actors.backend_for("gpt-5.6-sol", "high"))
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), FIXTURE_SHA256)
        self.assertEqual(seen["bundle_dirs"], [])


class SectionSplit(unittest.TestCase):

    def test_sections_partition_the_real_bundle_exactly(self):
        text = _real_context_text()
        sections = actor_context.split_sections(text)
        self.assertEqual("".join(s.text for s in sections), text)
        self.assertEqual([s.key for s in sections],
                         ["target", "program", "program_strategy", "profile", "already_tried",
                          "shared_history", "serving_observations", "inbox"])
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
            "prior_experiments": prior,
            "shared_prior_experiments": {"rows": [{"mechanism_id": "akm-s"}]},
            "serving_observations": {"rows": []},
            "prior_hypothesis_rejections": ["r1"], "prior_patch_rejections": ["r2"],
            "inbox": ["# note\n## Already tried\n- operator prose that repeats a header"],
        }
        text = actors.render_context(context)
        keys = [s.key for s in actor_context.split_sections(text)]
        for key in ("target", "program", "program_strategy", "superseded", "profile",
                    "exhausted_families", "stagnant_families", "already_tried",
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
        self.assertEqual(seen["env"], {actors.SEAT_ENV_ARM: "plain+ctx-variable"})
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
        self.assertEqual(hashlib.sha256(seen["prompt"].encode()).hexdigest(), FIXTURE_SHA256)
        self.assertIsNone(seen["env"], "an inline call is never recorded as the variable arm")


if __name__ == "__main__":
    unittest.main()
