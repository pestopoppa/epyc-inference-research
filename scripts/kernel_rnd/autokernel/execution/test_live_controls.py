import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from ..evaluator import statistics
from . import live_controls
from .. import campaign, schemas


class InstrumentSelection(unittest.TestCase):

    def test_missing_instrument_tools_are_built_as_complete_t0_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bindir = root / "build-v9-cpu" / "bin"
            bindir.mkdir(parents=True)
            def build(_argv, **_kwargs):
                for name in live_controls.INSTRUMENT_BUILD_TARGETS:
                    path = bindir / name
                    path.touch()
                    path.chmod(0o755)
            with mock.patch.object(live_controls, "INSTRUMENT_ROOT", root), \
                 mock.patch.object(live_controls, "INSTRUMENT_BINARY",
                                   bindir / "llama-bench"), \
                 mock.patch.object(live_controls.subprocess, "run", side_effect=build) as run:
                live_controls._ensure_instrument_build()
            run.assert_called_once_with(
                ["/usr/bin/cmake", "--build", str(root / "build-v9-cpu"), "--target",
                 "llama-completion", "llama-bench", "test-backend-ops", "-j", "64"],
                cwd=str(root), check=True)

    def tearDown(self):
        importlib.reload(live_controls)

    def test_clean_instrument_worktree_can_be_selected_explicitly(self):
        with mock.patch.dict(os.environ, {
                "AUTOKERNEL_INSTRUMENT_ROOT": "/tmp/ak-clean-instrument"}, clear=False):
            module = importlib.reload(live_controls)
        self.assertEqual(str(module.INSTRUMENT_ROOT), "/tmp/ak-clean-instrument")
        self.assertEqual(
            str(module.INSTRUMENT_BINARY),
            "/tmp/ak-clean-instrument/build-ak-t0-cpu-f744cc220/bin/llama-bench")

    def test_binary_override_is_separate_and_explicit(self):
        with mock.patch.dict(os.environ, {
                "AUTOKERNEL_INSTRUMENT_ROOT": "/tmp/ak-clean-instrument",
                "AUTOKERNEL_INSTRUMENT_BINARY": "/tmp/ak-build/bin/llama-bench",
        }, clear=False):
            module = importlib.reload(live_controls)
        self.assertEqual(str(module.INSTRUMENT_BINARY), "/tmp/ak-build/bin/llama-bench")


class InstrumentCapability(unittest.TestCase):

    def test_requires_every_binding_hardening_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-bench"
            binary.write_bytes(b"launcher")
            library = Path(tmp) / "libllama-bench-impl.so"
            library.write_bytes(b"\0".join(live_controls.REQUIRED_HARDENING_RECEIPTS))
            check = live_controls._instrument_receipt_capability(binary)
        self.assertEqual(check.outcome, "PASS")

    def test_names_missing_receipts_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-bench"
            binary.write_bytes(b"autokernel_hybrid_ab_complete")
            check = live_controls._instrument_receipt_capability(binary)
        self.assertEqual(check.outcome, "FAIL")
        self.assertIn("autokernel_device_sync_mode", check.reasons[0])

    def test_resume_linkage_identity_ignores_only_aslr_addresses(self):
        before = ("\tlibfoo.so => /sealed/libfoo.so (0x00007f000000)\n"
                  "\t/lib64/ld-linux-x86-64.so.2 (0x00007f100000)\n")
        after = ("\tlibfoo.so => /sealed/libfoo.so (0x00006a000000)\n"
                 "\t/lib64/ld-linux-x86-64.so.2 (0x00006a100000)\n")
        changed = after.replace("/sealed/libfoo.so", "/other/libfoo.so")
        self.assertEqual(
            live_controls._normalized_linkage(before),
            live_controls._normalized_linkage(after))
        self.assertNotEqual(
            live_controls._normalized_linkage(before),
            live_controls._normalized_linkage(changed))


class CampaignIdentity(unittest.TestCase):

    def test_distinct_campaigns_have_distinct_seeds_and_windows(self):
        first = live_controls.LiveCampaignIdentity(
            "ak-controls-v9-first", "/tmp/ak-controls-v9-first")
        second = live_controls.LiveCampaignIdentity(
            "ak-controls-v9-second", "/tmp/ak-controls-v9-second")
        self.assertNotEqual(first.campaign_seed, second.campaign_seed)
        self.assertNotEqual(first.window_id, second.window_id)
        self.assertTrue(first.window_id.startswith("akw-"))

    def test_evidence_reference_must_be_absolute(self):
        with self.assertRaisesRegex(ValueError, "absolute durable path"):
            live_controls.LiveCampaignIdentity(
                "ak-controls-v9-relative", "data/ak-controls-v9-relative")

    def test_parser_requires_and_preserves_fresh_identity(self):
        args = live_controls.build_parser().parse_args([
            "--campaign-id", "ak-controls-v9-parser",
            "--output", "/tmp/ak-controls-v9-parser",
        ])
        self.assertEqual(args.campaign_id, "ak-controls-v9-parser")
        self.assertEqual(args.output, Path("/tmp/ak-controls-v9-parser"))

    def test_parser_makes_live_and_existing_composition_exclusive(self):
        with self.assertRaises(SystemExit):
            live_controls.build_parser().parse_args([
                "--campaign-id", "ak-controls-v9-parser",
                "--output", "/tmp/ak-controls-v9-parser",
                "--execute", "--evaluate-existing",
            ])

    def test_parser_makes_resume_and_fresh_execution_exclusive(self):
        with self.assertRaises(SystemExit):
            live_controls.build_parser().parse_args([
                "--campaign-id", "ak-controls-v9-parser",
                "--output", "/tmp/ak-controls-v9-parser",
                "--execute", "--resume-existing",
            ])

    def test_current_source_claim_names_v9_and_exact_commit(self):
        reason = live_controls.CURRENT_SOURCE_CORRECTNESS_REASON
        self.assertIn("frozen v9 source", reason)
        self.assertIn(live_controls.PRODUCTION_COMMIT, reason)
        self.assertNotIn("frozen v8 source", reason)

    def test_existing_composition_cannot_overwrite_a_terminal_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                live_controls.evaluate_existing(
                    root, campaign_id="ak-controls-v9-terminal")


class CalibrationFrame(unittest.TestCase):

    def test_calibration_is_explicitly_iqk_off_at_campaign_repetition_count(self):
        """A/A must govern the same off-baseline that campaigns compare."""
        self.assertEqual(live_controls.CALIBRATION_REPS, 1)
        self.assertEqual(live_controls.CONTROL_ARM_IQK["aa_calibration"], ("0", "0"))
        self.assertEqual(
            live_controls.CONTROL_ARM_IQK["neutral_calibration"], ("0", "0"))
        self.assertEqual(
            live_controls._params(prompt=live_controls.PROMPT_TOKENS)["reps"], 1)

    def test_declaration_precommits_the_calibration_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = live_controls.LiveCampaignIdentity("ak-controls-frame", str(root))
            live_controls._write_declaration(
                root, identity=identity, instrument_sha="1" * 64,
                copy_sha="1" * 64, instrument_linkage="2" * 64,
                copy_linkage="2" * 64, toolchain_manifest_sha256="3" * 64,
                sealed_binding=live_controls.recipes.ToolBinding(
                    binary="/sealed/llama-bench", source_root="/sealed",
                    library_path="/sealed"))
            declaration = json.loads(
                (root / "campaign_declaration.json").read_text(encoding="utf-8"))
            source = json.loads(
                (root / "runtime-source-label.json").read_text(encoding="utf-8"))
        self.assertEqual(declaration["calibration_frame"], {
            "recipe_id": live_controls.RECIPE_ID,
            "prompt_tokens": live_controls.PROMPT_TOKENS,
            "reps": 1,
            "candidate_ggml_iqk": "0",
            "anchor_ggml_iqk": "0",
        })
        self.assertEqual(declaration["source_sha256"], source["source_sha256"])

    def test_declaration_precommits_a_non_ranked_campaign_length_anchor_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = live_controls.LiveCampaignIdentity("ak-controls-anchor-window", str(root))
            live_controls._write_declaration(
                root, identity=identity, instrument_sha="1" * 64,
                copy_sha="1" * 64, instrument_linkage="2" * 64,
                copy_linkage="2" * 64, toolchain_manifest_sha256="3" * 64,
                sealed_binding=live_controls.recipes.ToolBinding(
                    binary="/sealed/llama-bench", source_root="/sealed",
                    library_path="/sealed"))
            declaration = json.loads(
                (root / "campaign_declaration.json").read_text(encoding="utf-8"))
        self.assertEqual(declaration["anchor_motion_window_blocks"], 15)
        self.assertEqual(
            declaration["anchor_motion_settling"],
            live_controls.ANCHOR_MOTION_SETTLING)
        self.assertEqual(
            declaration["between_leg_policy"],
            live_controls.BETWEEN_LEG_POLICY)

    def test_measurement_plan_carries_the_declared_baseline_frame(self):
        """The executor must not silently fall back to recipe defaults."""
        class Run:
            complete = True

            @staticmethod
            def raw_vector():
                return {"fixture": True}

        captured = {}
        real_plan = live_controls.microbench.MicrobenchPlan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = live_controls.recipes.ToolBinding(
                binary="/sealed/llama-bench", source_root="/sealed", library_path="/sealed")
            worktree = live_controls.recipes.ToolBinding(
                binary="/worktree/llama-bench", source_root="/worktree", library_path="/worktree")
            anchor = live_controls.api.AnchorIdentity(
                source_commit="f" * 40, binary_sha256="1" * 64,
                linkage_sha256="2" * 64, tool="llama-bench")
            with mock.patch.object(
                    live_controls.microbench, "MicrobenchPlan",
                    side_effect=lambda **kwargs: captured.update(kwargs)
                    or real_plan(**kwargs)), \
                 mock.patch.object(
                    live_controls.microbench, "CpuRegionClaimAdapter"), \
                 mock.patch.object(
                    live_controls.microbench, "MicrobenchRunner") as runner:
                runner.return_value.run.return_value = Run()
                live_controls._measure(
                    label="aa_calibration", blocks=1, claim=object(),
                    candidate_binding=binding, anchor_binding=worktree, anchor=anchor,
                    candidate_iqk="0", anchor_iqk="0", output_root=root,
                    host_state=mock.Mock(), identity=live_controls.LiveCampaignIdentity(
                        "ak-controls-frame-run", str(root)))
                policy = runner.call_args.kwargs["policy"]
        self.assertEqual(captured["params"]["reps"], 1)
        self.assertEqual(captured["candidate_param_overrides"], {"ggml_iqk": "0"})
        self.assertEqual(captured["anchor_param_overrides"], {"ggml_iqk": "0"})
        self.assertEqual(captured["candidate_binding"], binding)
        self.assertEqual(captured["anchor_binding"], binding)
        self.assertEqual(captured["pairs_per_block"], 1)
        self.assertFalse(policy.require_load)
        self.assertTrue(policy.require_package_power)

    def test_decode_recipe_has_its_own_non_prefill_frame(self):
        """Decode calibration must never silently reuse pp512 inputs."""
        try:
            live_controls.configure_recipe(live_controls.DECODE_RECIPE_ID)
            self.assertEqual(live_controls.RECIPE_ID, live_controls.DECODE_RECIPE_ID)
            self.assertEqual(live_controls._params(prompt=live_controls.PROMPT_TOKENS)["n_gen"],
                             live_controls.DECODE_TOKENS)
            self.assertNotIn("n_prompt", live_controls._params(
                prompt=live_controls.PROMPT_TOKENS))
            self.assertIn(":tg128:", live_controls._unit_id(
                label="aa_calibration", prompt=live_controls.PROMPT_TOKENS))
        finally:
            importlib.reload(live_controls)

    def test_decode_measure_resolves_default_frame_at_execution_time(self):
        """The execute path must not retain pp512 from function definition."""
        class Run:
            complete = True

            @staticmethod
            def raw_vector():
                return {"fixture": True}

        captured = {}
        real_plan = live_controls.microbench.MicrobenchPlan
        try:
            live_controls.configure_recipe(live_controls.DECODE_RECIPE_ID)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                binding = live_controls.recipes.ToolBinding(
                    binary="/sealed/llama-bench", source_root="/sealed",
                    library_path="/sealed")
                anchor = live_controls.api.AnchorIdentity(
                    source_commit="f" * 40, binary_sha256="1" * 64,
                    linkage_sha256="2" * 64, tool="llama-bench")
                with mock.patch.object(
                        live_controls.microbench, "MicrobenchPlan",
                        side_effect=lambda **kwargs: captured.update(kwargs)
                        or real_plan(**kwargs)), \
                     mock.patch.object(
                        live_controls.microbench, "CpuRegionClaimAdapter"), \
                     mock.patch.object(
                        live_controls.microbench, "MicrobenchRunner") as runner:
                    runner.return_value.run.return_value = Run()
                    # Deliberately omit ``prompt`` exactly as execute() does for
                    # every committed-cell control.
                    live_controls._measure(
                        label="aa_calibration", blocks=1, claim=object(),
                        candidate_binding=binding, anchor_binding=binding,
                        anchor=anchor, candidate_iqk="0", anchor_iqk="0",
                        output_root=root, host_state=mock.Mock(),
                        identity=live_controls.LiveCampaignIdentity(
                            "ak-controls-decode-run", str(root)))
            self.assertEqual(captured["recipe_id"], live_controls.DECODE_RECIPE_ID)
            self.assertEqual(captured["params"]["n_gen"], 128)
            self.assertNotIn("n_prompt", captured["params"])
            self.assertIn(":tg128:", captured["unit_ids"][0])
            self.assertEqual(
                captured["pairs_per_block"],
                live_controls.DECODE_FRESH_PAIRS_PER_BLOCK)
        finally:
            importlib.reload(live_controls)

    def test_parser_selects_decode_before_any_execution(self):
        args = live_controls.build_parser().parse_args([
            "--campaign-id", "ak-controls-v9-decode", "--output", "/tmp/ak-decode",
            "--recipe", live_controls.DECODE_RECIPE_ID])
        self.assertEqual(args.recipe, live_controls.DECODE_RECIPE_ID)

    def test_decode_declaration_binds_decode_frame(self):
        try:
            live_controls.configure_recipe(live_controls.DECODE_RECIPE_ID)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                identity = live_controls.LiveCampaignIdentity("ak-controls-decode", str(root))
                live_controls._write_declaration(
                    root, identity=identity, instrument_sha="1" * 64,
                    copy_sha="1" * 64, instrument_linkage="2" * 64,
                    copy_linkage="2" * 64, toolchain_manifest_sha256="3" * 64,
                    sealed_binding=live_controls.recipes.ToolBinding(
                        binary="/sealed/llama-bench", source_root="/sealed",
                        library_path="/sealed"))
                declaration = json.loads((root / "campaign_declaration.json").read_text())
            self.assertEqual(declaration["recipe_id"], live_controls.DECODE_RECIPE_ID)
            self.assertEqual(declaration["calibration_frame"]["decode_tokens"], 128)
            self.assertNotIn("prompt_tokens", declaration["calibration_frame"])
            self.assertEqual(
                declaration["calibration_frame"]["fresh_pairs_per_block"], 5)
            self.assertEqual(
                declaration["calibration_frame"]["aggregation"], "median_per_arm")
        finally:
            importlib.reload(live_controls)

    def test_decode_dry_run_discloses_fresh_invocation_cost(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = live_controls.main([
                "--campaign-id", "ak-controls-decode-dry",
                "--output", "/durable/decode-dry",
                "--recipe", live_controls.DECODE_RECIPE_ID,
            ])
        plan = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(plan["calibration_frame"]["fresh_pairs_per_block"], 5)
        self.assertEqual(plan["calibration_frame"]["reps"], 1)
        self.assertEqual(plan["calibration_fresh_invocations"], 2600)


class BetweenLegPolicy(unittest.TestCase):

    @staticmethod
    def _attestation():
        return live_controls.microbench.ClaimAttestation(
            claim_id="akclaim-test", holder="test-holder",
            cpu_list=live_controls.CPU_LIST,
            observed_at="2026-08-13T00:00:00+00:00",
            check=schemas.Check(schemas.PASS))

    def test_high_ordinary_load_is_recorded_and_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = mock.Mock()
            adapter.attest.return_value = self._attestation()
            original_read_text = Path.read_text

            def read_text(path, *args, **kwargs):
                if str(path) == "/proc/loadavg":
                    return "999.00 999.00 999.00 1/1 1\n"
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(
                    live_controls.microbench, "CpuRegionClaimAdapter",
                    return_value=adapter), \
                 mock.patch.object(
                    live_controls.screening_baseline, "competing_inference_witness",
                    return_value={
                        "basis": "interim_inference_executable_scan",
                        "competing": False, "findings": [],
                        "ordinary_processes_ignored": True,
                    }), \
                 mock.patch.object(Path, "read_text", autospec=True,
                                   side_effect=read_text):
                receipt = live_controls._observe_between_legs(
                    root, boundary="aa_to_neutral", claim=object())
            self.assertEqual(receipt["ordinary_load"]["load1"], 999.0)
            self.assertEqual(
                receipt["ordinary_load"]["disposition"],
                "recorded_as_noise_not_a_gate")
            self.assertFalse(receipt["inference_witness"]["competing"])

    def test_competing_llama_is_recorded_and_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = mock.Mock()
            adapter.attest.return_value = self._attestation()
            with mock.patch.object(
                    live_controls.microbench, "CpuRegionClaimAdapter",
                    return_value=adapter), \
                 mock.patch.object(
                    live_controls.screening_baseline, "competing_inference_witness",
                    return_value={
                        "basis": "interim_inference_executable_scan",
                        "competing": True,
                        "findings": [{"argv0_basename": "llama-server"}],
                        "ordinary_processes_ignored": True,
                    }):
                with self.assertRaisesRegex(RuntimeError, "competing model inference"):
                    live_controls._observe_between_legs(
                        root, boundary="aa_to_neutral", claim=object())
            records = [json.loads(line) for line in (
                root / "between_leg_observations.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["inference_witness"]["competing"])
            body = dict(records[0])
            self.assertEqual(
                body.pop("observation_sha256"), schemas.content_hash(body))


class RecordedCompositionMaterial(unittest.TestCase):

    def test_canonical_paired_block_round_trips(self):
        block = statistics.PairedBlock(
            block_index=3, unit_id="fixture:pp512", stratum="selection",
            order="candidate_first", anchor_samples=(1.0,),
            candidate_samples=(1.1,), measured_at="2026-08-12T00:00:00+00:00")
        self.assertEqual(
            live_controls._paired_block_from_raw(block.to_list()), block)

    def test_noncanonical_paired_block_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nine-field"):
            live_controls._paired_block_from_raw([0, "too-short"])


class CalibrationBlockCheckpoint(unittest.TestCase):

    @staticmethod
    def _plan(*, blocks=4, seed="checkpoint-seed"):
        binding = live_controls.recipes.ToolBinding(
            binary="/sealed/llama-bench", source_root="/sealed",
            library_path="/sealed")
        return live_controls.microbench.MicrobenchPlan(
            recipe_id=live_controls.PREFILL_RECIPE_ID,
            candidate_id="akc-control-aa_calibration", campaign_seed=seed,
            candidate_binding=binding, anchor_binding=binding,
            anchor=live_controls.api.AnchorIdentity(
                source_commit="f" * 40, binary_sha256="1" * 64,
                linkage_sha256="2" * 64, tool="llama-bench"),
            params={"model": "/sealed/model.gguf", "n_prompt": 512,
                    "reps": 1, "autokernel_seed": 1,
                    "output_format": "json"},
            base_blocks=blocks, pairs_per_block=1,
            unit_ids=("model.gguf:pp512:aa_calibration",))

    @staticmethod
    def _record(block_plan, plan):
        invocations = []
        anchor_samples = []
        candidate_samples = []
        for position, arm in enumerate(block_plan.arm_sequence):
            value = float(100 + block_plan.block_index * 10 + position)
            (anchor_samples if arm == live_controls.microbench.ARM_ANCHOR
             else candidate_samples).append(value)
            binding = (plan.candidate_binding
                       if arm == live_controls.microbench.ARM_CANDIDATE
                       else plan.anchor_binding)
            params = plan.params_for(arm, block_plan.unit_id)
            argv = ("fixture-llama-bench", arm)
            registry_id = "fixture-registry/v1"
            recipe_env = {}
            env = {}
            receipt = live_controls.microbench.ExecutionReceipt(
                runner_id="fixture/v1", recipe_id=plan.recipe_id,
                registry_id=registry_id, arm=arm,
                constructor_id="fixture-constructor/v1",
                constructor_sha256="3" * 64,
                argv_sha256=live_controls.microbench._argv_hash(
                    recipe_id=plan.recipe_id, registry_id=registry_id,
                    arm=arm, argv=argv, env=recipe_env, params=params),
                argv=argv, recipe_env=recipe_env, params=params, env=env,
                env_sha256=live_controls.microbench._env_hash(env),
                binary_path=binding.binary, binary_sha256="4" * 64,
                binary_size=1, source_root=binding.source_root,
                library_path=binding.library_path,
                resolved_at="2026-08-13T00:00:00+00:00")
            invocations.append({
                "block_index": block_plan.block_index, "position": position,
                "arm": arm, "recipe": receipt.render(),
                "receipt": receipt.to_dict(),
                "spawn": {"stdout_sha256": str(position),
                           "sandbox_receipt": None},
                "row": {"samples_ts": [value]}, "samples": [value],
                "claim": {
                    "claim_id": "claim-old", "holder": "pid:1",
                    "cpu_list": live_controls.CPU_LIST,
                    "observed_at": "2026-08-13T00:00:00+00:00",
                    "outcome": schemas.PASS, "reasons": [],
                },
                "checks": [],
            })
        paired = statistics.PairedBlock(
            block_index=block_plan.block_index, unit_id=block_plan.unit_id,
            stratum=block_plan.stratum, order=block_plan.order,
            segment=block_plan.segment,
            extension_round=block_plan.extension_round,
            measured_at="2026-08-13T00:00:01+00:00",
            anchor_samples=tuple(anchor_samples),
            candidate_samples=tuple(candidate_samples))
        raw = {
            "plan": block_plan.to_dict(), "invocations": invocations,
            "host_state_open": {}, "host_state_close": {},
            "package_power": {}, "paired_block": paired.to_list(),
            "checks": [], "refusals": [], "complete": True,
        }

        class Record:
            plan = block_plan

            @staticmethod
            def to_dict():
                return raw

        return Record()

    def _checkpoint(self, root, *, plan=None):
        plan = plan or self._plan()
        identity = live_controls.LiveCampaignIdentity(
            "ak-controls-checkpoint", str(root))
        return live_controls._BlockCheckpoint(
            root, identity=identity, label="aa_calibration", plan=plan)

    def test_interrupted_prefix_reopens_at_exact_next_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan()
            planned = live_controls.microbench.plan_blocks(
                plan.schedule(), count=plan.blocks_to_run,
                pairs=plan.pairs_per_block, unit_ids=plan.unit_ids,
                stratum=plan.stratum)
            checkpoint = self._checkpoint(root, plan=plan)
            checkpoint.append(self._record(planned[0], plan))
            checkpoint.append(self._record(planned[1], plan))
            resumed = self._checkpoint(root, plan=plan)
        self.assertEqual(len(resumed.prefix), 2)
        self.assertEqual(resumed.prefix[-1].plan.block_index, 1)
        self.assertEqual(resumed.head_sha256, checkpoint.head_sha256)

    def test_tampered_block_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan()
            planned = live_controls.microbench.plan_blocks(
                plan.schedule(), count=plan.blocks_to_run,
                pairs=1, unit_ids=plan.unit_ids, stratum=plan.stratum)
            checkpoint = self._checkpoint(root, plan=plan)
            checkpoint.append(self._record(planned[0], plan))
            path = checkpoint.blocks_path
            entry = json.loads(path.read_text())
            entry["block"]["invocations"][0]["samples"][0] += 1
            path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash chain"):
                self._checkpoint(root, plan=plan)

    def test_consistently_rehashed_order_mutation_still_fails_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan()
            planned = live_controls.microbench.plan_blocks(
                plan.schedule(), count=plan.blocks_to_run,
                pairs=1, unit_ids=plan.unit_ids, stratum=plan.stratum)
            checkpoint = self._checkpoint(root, plan=plan)
            checkpoint.append(self._record(planned[0], plan))
            path = checkpoint.blocks_path
            entry = json.loads(path.read_text())
            entry["block"]["plan"]["order"] = (
                statistics.ORDER_CANDIDATE_FIRST
                if planned[0].order == statistics.ORDER_ANCHOR_FIRST
                else statistics.ORDER_ANCHOR_FIRST)
            entry["block_sha256"] = schemas.content_hash(entry["block"])
            body = {key: value for key, value in entry.items()
                    if key != "entry_sha256"}
            entry["entry_sha256"] = schemas.content_hash(body)
            path.write_text(schemas.canonical_json(entry) + "\n", encoding="utf-8")
            with self.assertRaises(live_controls.microbench.ScheduleMismatch):
                self._checkpoint(root, plan=plan)

    def test_manifest_refuses_frame_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._checkpoint(root, plan=self._plan(seed="first"))
            with self.assertRaisesRegex(ValueError, "campaign/frame"):
                self._checkpoint(root, plan=self._plan(seed="second"))


class CalibrationBoundaryResume(unittest.TestCase):

    def test_claim_budget_covers_the_declared_long_decode_campaign(self):
        self.assertEqual(live_controls.CLAIM_MAX_HOLD_S, 6 * 3600)

    def test_derived_calibration_is_admitted_only_when_recomputed_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            for name in ("aa_calibration.json", "neutral_calibration.json"):
                (root / "raw" / name).write_text("{}\n", encoding="utf-8")
            expected = {"accepted": True, "derived": "fixture"}
            (root / "calibration.json").write_text(
                json.dumps(expected) + "\n", encoding="utf-8")
            material = mock.Mock()
            solve = mock.Mock()
            solve.to_dict.return_value = expected
            identity = live_controls.LiveCampaignIdentity(
                "ak-controls-calibration-boundary", str(root))
            # Stop immediately after the new boundary validation.  This keeps
            # the test narrow while proving calibration.json is no longer a
            # blanket terminal-material refusal and is recomputed, not trusted.
            with mock.patch.object(
                    live_controls, "_load_recorded_material",
                    return_value=(material, {})) as load, \
                 mock.patch.object(
                    live_controls, "_campaign_inputs",
                    return_value=(None, None, None, None, solve)), \
                 mock.patch.object(
                    live_controls, "_load_json",
                    side_effect=lambda path: (
                        expected if Path(path).name == "calibration.json"
                        else (_ for _ in ()).throw(RuntimeError("boundary passed")))):
                with self.assertRaisesRegex(RuntimeError, "boundary passed"):
                    live_controls._validate_resume_existing(root, identity=identity)
            self.assertEqual(load.call_count, 2)

    def test_tampered_derived_calibration_fails_before_resume_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            for name in ("aa_calibration.json", "neutral_calibration.json"):
                (root / "raw" / name).write_text("{}\n", encoding="utf-8")
            (root / "calibration.json").write_text(
                json.dumps({"accepted": False}) + "\n", encoding="utf-8")
            solve = mock.Mock()
            solve.to_dict.return_value = {"accepted": True}
            identity = live_controls.LiveCampaignIdentity(
                "ak-controls-calibration-tamper", str(root))
            with mock.patch.object(
                    live_controls, "_load_recorded_material",
                    return_value=(mock.Mock(), {})), \
                 mock.patch.object(
                    live_controls, "_campaign_inputs",
                    return_value=(None, None, None, None, solve)):
                with self.assertRaisesRegex(ValueError, "deterministic solve"):
                    live_controls._validate_resume_existing(root, identity=identity)


class ProspectiveBeliefReceipt(unittest.TestCase):

    class _Result:
        may_rank = True

        def to_dict(self):
            controls = (
                "positive", "neutral", "degraded_negative", "aa",
                "historical_win_replay")
            return {
                "panel_result": {
                    "marker": "5/5", "may_rank": True,
                    "halts_campaign": False, "voids_window": False,
                    "outcomes": [{
                        "control_id": control, "ordinal": index,
                        "outcome": "PASS", "disposition": "satisfied",
                    } for index, control in enumerate(controls, 1)],
                    "observations": [{
                        "control_id": control, "abs_effect_count": 15,
                        "effect_resolution": (
                            "improvement" if control in {
                                "positive", "historical_win_replay"}
                            else "below_noise_floor"),
                    } for control in controls],
                },
            }

    class _Claim:
        def to_dict(self):
            return {
                "schema": "epyc.autokernel.cpu_region_claim_receipt.v1",
                "claim_id": "akc-live-claim", "campaign_id": "ak-live-next",
                "cpu_list": live_controls.CPU_LIST,
                "acquired_at": "2026-08-12T02:00:00+00:00",
                "released_at": "2026-08-12T02:10:00+00:00",
            }

    def _fixture(self, root: Path, *, prospective: bool = True):
        runtime_body = {
            "schema": "epyc.autokernel.runtime_source_label.v1",
            "production_source_commit": live_controls.PRODUCTION_COMMIT,
            "measurement_instrument_commit": live_controls.INSTRUMENT_COMMIT,
            "measurement_binary_sha256": "1" * 64,
            "copied_binary_sha256": "1" * 64,
            "measurement_linkage_sha256": "2" * 64,
            "copied_linkage_sha256": "2" * 64,
            "binary_copy_exact": True,
        }
        declaration = {
            "schema": "epyc.autokernel.live_control_campaign_declaration.v1",
            "campaign_id": "ak-live-next", "model": "/models/tiny.gguf",
            "model_sha256": "3" * 64,
        }
        if prospective:
            declaration["belief_capture_schema"] = live_controls.BELIEF_RECEIPT_SCHEMA
        for name, value in (
                ("campaign_declaration.json", declaration),
                ("runtime-source-label.json", {
                    **runtime_body, "source_sha256": schemas.content_hash(runtime_body)}),
                ("control_sweep.json", self._Result().to_dict())):
            (root / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
        for label in (
                "positive", "neutral_calibration", "negative_committed_cell",
                "negative_wrong_cell", "aa_calibration", "historical_win_replay"):
            path = root / "raw" / f"{label}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"label": label}) + "\n", encoding="utf-8")

    def test_future_writer_emits_five_self_hashed_identity_bound_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            receipt = live_controls._build_belief_receipt(
                root, identity=live_controls.LiveCampaignIdentity(
                    "ak-live-next", str(root)), result=self._Result(),
                claim_receipt=self._Claim(),
                measured_at="2026-08-12T02:10:00+00:00")
        self.assertIsNotNone(receipt)
        self.assertEqual(len(receipt["belief_measurements"]), 5)
        self.assertEqual(receipt["native_verdict"]["marker"], "5/5")
        for row in receipt["belief_measurements"]:
            self.assertEqual(row["protocol_id"], "P-AK-SEARCH-1/v1")
            self.assertEqual(row["reps_basis"], "scored:paired live-control blocks")
            self.assertEqual(
                row["extra"]["evidence_sha256"],
                schemas.content_hash(row["extra"]["evidence_basis"]))
            unsigned = dict(row)
            stored = unsigned.pop("measurement_sha256")
            self.assertEqual(stored, schemas.content_hash(unsigned))
        unsigned_receipt = dict(receipt)
        stored_receipt = unsigned_receipt.pop("receipt_sha256")
        self.assertEqual(stored_receipt, schemas.content_hash(unsigned_receipt))

    def test_pre_hook_declaration_is_not_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root, prospective=False)
            receipt = live_controls._build_belief_receipt(
                root, identity=live_controls.LiveCampaignIdentity(
                    "ak-live-next", str(root)), result=self._Result(),
                claim_receipt=self._Claim(),
                measured_at="2026-08-12T02:10:00+00:00")
        self.assertIsNone(receipt)


class ControlEffectReachability(unittest.TestCase):

    def test_predeclared_control_window_crosses_where_base_segment_cannot(self):
        rule = live_controls._control_stopping_rule()
        self.assertEqual(rule.max_total_blocks(5), 10)
        construction = statistics.select_construction(
            "sign_martingale_predictable_lambda/v1")
        base = statistics.run_e_process(
            (0.08,) * 5, construction=construction,
            hypothesis=statistics.HYPOTHESIS_IMPROVEMENT,
            margin=0.0, threshold=10.0)
        full = statistics.run_e_process(
            (0.08,) * rule.max_total_blocks(5), construction=construction,
            hypothesis=statistics.HYPOTHESIS_IMPROVEMENT,
            margin=0.0, threshold=10.0)
        self.assertLess(base.e_running_max, 10.0)
        self.assertGreaterEqual(full.e_running_max, 10.0)
        self.assertEqual(full.first_crossing_block, 7)


if __name__ == "__main__":
    unittest.main()
