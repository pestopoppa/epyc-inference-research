import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k35_stack_context_matrix_runner as k35


class K35StackContextMatrixRunnerTests(unittest.TestCase):
    def test_worker_command_preserves_composed_spec_knobs(self):
        scenario = k35.scenario_by_name("worker_general_cpu_composed_spec")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/k35-build/bin/llama-server"),
            port=19123,
            nominal_context=2048,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("LD_LIBRARY_PATH=/tmp/k35-build/bin", argv)
        self.assertIn("ROCR_VISIBLE_DEVICES=-1", argv)
        self.assertIn("HIP_VISIBLE_DEVICES=-1", argv)
        self.assertIn("CUDA_VISIBLE_DEVICES=", argv)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=0", argv)
        self.assertNotIn("HIP_VISIBLE_DEVICES=0", argv)
        self.assertIn("--spec-type ngram-mod,draft-mtp", joined)
        self.assertIn("--spec-draft-n-max 2", joined)
        self.assertIn("--spec-draft-threads 16", joined)
        self.assertIn("--spec-draft-device none", joined)
        self.assertIn("--no-mmap", argv)
        self.assertIn("--no-op-offload", argv)
        self.assertIn("--no-kv-offload", argv)
        self.assertIn("-ctk q8_0 -ctv q8_0", joined)

    def test_frontdoor_command_uses_gpu_no_spec(self):
        scenario = k35.scenario_by_name("frontdoor_gpu_resident_no_spec")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19124,
            nominal_context=8192,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("ROCR_VISIBLE_DEVICES=0", argv)
        self.assertIn("HIP_VISIBLE_DEVICES=0", argv)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", argv)
        self.assertIn("--device ROCm0", joined)
        self.assertIn("-ngl 99", joined)
        self.assertIn("--spec-type none", joined)
        self.assertIn("--reasoning off", joined)
        self.assertEqual(scenario.enable_thinking, False)

    def test_frontdoor_cpu_anchor_uses_no_gpu_no_spec(self):
        scenario = k35.scenario_by_name("frontdoor_cpu_no_spec")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19124,
            nominal_context=8192,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("ROCR_VISIBLE_DEVICES=-1", argv)
        self.assertIn("HIP_VISIBLE_DEVICES=-1", argv)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=0", argv)
        self.assertNotIn("HIP_VISIBLE_DEVICES=0", argv)
        self.assertIn("--device none", joined)
        self.assertIn("-ngl 0", joined)
        self.assertIn("--spec-type none", joined)
        self.assertIn("-ctk q8_0 -ctv q8_0", joined)
        self.assertEqual(scenario.enable_thinking, False)

    def test_frontdoor_gpu_native_mtp_uses_same_file_draft(self):
        scenario = k35.scenario_by_name("frontdoor_gpu_native_mtp")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19124,
            nominal_context=8192,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("--device ROCm0", joined)
        self.assertIn("-ngl 99", joined)
        self.assertIn("--spec-type draft-mtp", joined)
        self.assertIn("--spec-draft-n-max 3", joined)
        self.assertNotIn("-md", argv)
        self.assertEqual(scenario.enable_thinking, False)

    def test_architect_command_preserves_native_mtp_and_thinking_off(self):
        scenario = k35.scenario_by_name("architect_general_cpu_native_mtp")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19125,
            nominal_context=2048,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("-np 2", joined)
        self.assertIn("--spec-type draft-mtp", joined)
        self.assertIn("--spec-draft-n-max 4", joined)
        self.assertIn("-ctk q4_0 -ctv f16", joined)
        self.assertIn("--mlock", argv)
        self.assertIn("--slot-save-path /mnt/raid0/llm/cache/kv_slots/architect_general", joined)
        self.assertNotIn("-md", argv)
        self.assertEqual(scenario.enable_thinking, False)

    def test_architect_context_accounts_for_parallel_slots(self):
        scenario = k35.scenario_by_name("architect_general_cpu_native_mtp")
        self.assertEqual(
            k35.server_context(scenario, nominal_context=8192, max_tokens=512),
            16384,
        )

    def test_ingest_command_uses_default_experts_without_spec(self):
        scenario = k35.scenario_by_name("ingest_long_context_cpu_default_experts")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/llama-server"),
            port=19126,
            nominal_context=8192,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("--spec-type none", joined)
        self.assertNotIn("--override-kv", argv)
        self.assertNotIn("qwen3next.expert_used_count", joined)
        self.assertIn("-ctk q4_0 -ctv q4_0", joined)
        self.assertIn("--mlock", argv)

    def test_plan_skips_contexts_above_scenario_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = k35.parse_args(
                [
                    "--only",
                    "worker_general_cpu_composed_spec",
                    "--context",
                    "2048",
                    "--context",
                    "32768",
                    "--max-tokens",
                    "128",
                    "--output-dir",
                    tmp,
                ]
            )
            plan = k35.build_plan(args)
            self.assertEqual([cell["nominal_context"] for cell in plan["cells"]], [2048, 32768])

    def test_plan_skips_architect_contexts_above_per_slot_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = k35.parse_args(
                [
                    "--only",
                    "architect_general_cpu_native_mtp",
                    "--context",
                    "8192",
                    "--context",
                    "14000",
                    "--max-tokens",
                    "128",
                    "--output-dir",
                    tmp,
                ]
            )
            plan = k35.build_plan(args)
            self.assertEqual([cell["nominal_context"] for cell in plan["cells"]], [8192])

    def test_plan_expands_reps_with_unique_ports(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = k35.parse_args(
                [
                    "--only",
                    "frontdoor_gpu_resident_no_spec",
                    "--context",
                    "8192",
                    "--reps",
                    "3",
                    "--output-dir",
                    tmp,
                ]
            )
            plan = k35.build_plan(args)
            self.assertEqual(plan["reps"], 3)
            self.assertEqual([cell["rep"] for cell in plan["cells"]], [1, 2, 3])
            self.assertEqual([cell["nominal_context"] for cell in plan["cells"]], [8192, 8192, 8192])
            self.assertEqual(len({cell["port"] for cell in plan["cells"]}), 3)

    def test_plan_records_pgpu1_policy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = k35.parse_args(
                [
                    "--only",
                    "frontdoor_gpu_resident_no_spec",
                    "--context",
                    "8192",
                    "--warmup-discard-policy",
                    "no warm-up; no discard",
                    "--cpu-interference-policy",
                    "CPU stack quiesced",
                    "--output-dir",
                    tmp,
                ]
            )
            plan = k35.build_plan(args)
            fields = plan["pgpu1_protocol_fields"]
            self.assertEqual(fields["warmup_discard_policy"], "no warm-up; no discard")
            self.assertEqual(fields["cpu_interference_policy"], "CPU stack quiesced")
            self.assertIn("after_cleanup", fields["post_cleanup_vram_sample"])
            self.assertIn("before_launch", fields["pre_launch_gpu_sample"])
            self.assertIn("request.json", fields["request_artifacts"])
            self.assertIn("clocks", fields["rocm_hardware_state"])
            self.assertIn("--execute", plan["operator_invocation"])

    def test_build_chat_request_body_records_thinking_contract(self):
        scenario = k35.scenario_by_name("frontdoor_gpu_native_mtp")
        body = k35.build_chat_request_body(scenario, "Return exactly: OK", max_tokens=32)

        self.assertEqual(body["messages"][0]["content"], "Return exactly: OK")
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["seed"], 35)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_v9_frontdoor_matches_current_native_mtp_shape(self):
        scenario = k35.scenario_by_name("v9_frontdoor_cpu_native_mtp")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/v9-cpu/bin/llama-server"),
            port=19130,
            nominal_context=2048,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertIn("-np 4", joined)
        self.assertIn("-ub 8192", joined)
        self.assertIn("--device none", joined)
        self.assertIn("--spec-type draft-mtp", joined)
        self.assertIn("--spec-draft-n-max 4", joined)
        self.assertIn("--slot-save-path /mnt/raid0/llm/cache/kv_slots/frontdoor", joined)

    def test_v9_coder_alias_disables_speculation_per_request(self):
        scenario = k35.scenario_by_name("v9_coder_escalation_gpu_request_no_spec")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/v9-hip/bin/llama-server"),
            port=19131,
            nominal_context=2048,
            max_tokens=256,
        )
        body = k35.build_chat_request_body(scenario, "Return OK", max_tokens=32)
        joined = " ".join(argv)
        self.assertIn("--spec-type draft-mtp", joined)
        self.assertIn("--spec-draft-n-max 4", joined)
        self.assertEqual(body["speculative.n_max"], 0)

    def test_v9_vision_omits_undeclared_ubatch_and_adds_projector(self):
        scenario = k35.scenario_by_name("v9_worker_vision_gpu_no_spec")
        argv = k35.build_server_argv(
            scenario,
            binary=Path("/tmp/v9-hip/bin/llama-server"),
            port=19132,
            nominal_context=2048,
            max_tokens=256,
        )
        joined = " ".join(argv)
        self.assertNotIn("-ub", argv)
        self.assertIn("--mmproj", argv)
        self.assertIn("mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf", joined)
        self.assertIn("--image-min-tokens 1024", joined)

    def test_v9_dspark_pair_differs_only_at_request_cap(self):
        disabled = k35.scenario_by_name("v9_dsv4_q8_dspark_request_nmax0")
        enabled = k35.scenario_by_name("v9_dsv4_q8_dspark_request_nmax3")
        disabled_argv = k35.build_server_argv(
            disabled,
            binary=Path("/tmp/v9-cpu/bin/llama-server"),
            port=19133,
            nominal_context=2048,
            max_tokens=16,
        )
        enabled_argv = k35.build_server_argv(
            enabled,
            binary=Path("/tmp/v9-cpu/bin/llama-server"),
            port=19133,
            nominal_context=2048,
            max_tokens=16,
        )
        self.assertEqual(disabled_argv, enabled_argv)
        self.assertIn("--spec-type", disabled_argv)
        self.assertIn("draft-dspark", disabled_argv)
        self.assertEqual(
            k35.build_chat_request_body(disabled, "prompt", max_tokens=16)["speculative.n_max"],
            0,
        )
        self.assertEqual(
            k35.build_chat_request_body(enabled, "prompt", max_tokens=16)["speculative.n_max"],
            3,
        )
        self.assertEqual(
            k35.build_chat_request_body(disabled, "prompt", max_tokens=16)["prompt"],
            "prompt",
        )
        self.assertTrue(
            k35.build_chat_request_body(disabled, "prompt", max_tokens=16)["return_tokens"]
        )

    def test_v9_iq3_xxs_dspark_pair_uses_downloaded_target(self):
        disabled = k35.scenario_by_name("v9_dsv4_iq3_xxs_dspark_request_nmax0")
        enabled = k35.scenario_by_name("v9_dsv4_iq3_xxs_dspark_request_nmax3")
        self.assertEqual(disabled.model, enabled.model)
        self.assertIn("UD-IQ3_XXS", str(disabled.model))
        self.assertEqual(disabled.draft_model, enabled.draft_model)
        self.assertEqual(disabled.request_spec_n_max, 0)
        self.assertEqual(enabled.request_spec_n_max, 3)

    def test_dspark_summary_captures_tokens_and_effective_cap(self):
        scenario = k35.scenario_by_name("v9_dsv4_q8_dspark_request_nmax0")
        result = k35.summarize_response(
            scenario,
            2048,
            16,
            {
                "tokens": [1, 2, 3],
                "generation_settings": {"speculative.n_max": 0},
                "timings": {"predicted_n": 3, "draft_n": 0, "draft_n_accepted": 0},
            },
            1.0,
            3,
        )
        self.assertEqual(result["token_ids"], [1, 2, 3])
        self.assertEqual(result["effective_speculative_n_max"], 0)

    def test_dspark_parity_requires_caps_activity_and_exact_tokens(self):
        base = {"nominal_context": 2048, "rep": 1, "status": "ok"}
        result = k35.evaluate_dspark_parity(
            [
                {
                    **base,
                    "scenario": "v9_dsv4_q8_dspark_request_nmax0",
                    "effective_speculative_n_max": 0,
                    "draft_n": None,
                    "token_ids": [1, 2, 3],
                },
                {
                    **base,
                    "scenario": "v9_dsv4_q8_dspark_request_nmax3",
                    "effective_speculative_n_max": 3,
                    "draft_n": 4,
                    "token_ids": [1, 2, 3],
                },
            ]
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["comparisons"][0]["variant"], "q8")
        self.assertTrue(all(result["comparisons"][0]["checks"].values()))

    def test_main_dry_run_writes_plan_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = k35.main(
                [
                    "--only",
                    "frontdoor_gpu_resident_no_spec",
                    "--context",
                    "2048",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            plan = json.loads((Path(tmp) / "plan.json").read_text())
            self.assertEqual(len(plan["cells"]), 1)
            self.assertEqual(plan["reps"], 1)
            self.assertTrue((Path(tmp) / "commands.sh").exists())
            operator_run = Path(tmp) / "operator_run.sh"
            self.assertTrue(operator_run.exists())
            operator_text = operator_run.read_text()
            self.assertIn("--execute", operator_text)
            self.assertIn("--output-dir", operator_text)
            self.assertIn("K35_RUN_ID:=k35_stack_context_matrix_$(date -u +%Y%m%dT%H%M%SZ)", operator_text)
            self.assertIn(str(k35.RESEARCH_ROOT / "data/k35_stack_context_matrix"), operator_text)
            self.assertIn("P-GPU-1 caveat", operator_text)
            self.assertIn("production-named-kernel only", operator_text)
            self.assertNotIn(f"--output-dir {tmp}", operator_text)
            self.assertIn("production-named-kernel only", plan["pgpu1_protocol_fields"]["certification_note"])
            self.assertEqual(
                plan["pgpu1_protocol_fields"]["operator_execution_output_dir"],
                "${K35_EXECUTION_BASE}/${K35_RUN_ID}",
            )

    def test_operator_run_static_execution_output_dir_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prep"
            execution_dir = Path(tmp) / "execute"
            rc = k35.main(
                [
                    "--only",
                    "frontdoor_gpu_resident_no_spec",
                    "--context",
                    "2048",
                    "--output-dir",
                    str(prep_dir),
                    "--execution-output-dir",
                    str(execution_dir),
                ]
            )

            self.assertEqual(rc, 0)
            operator_text = (prep_dir / "operator_run.sh").read_text()
            self.assertIn(f'K35_EXEC_OUTPUT_DIR:-{execution_dir.resolve()}', operator_text)
            self.assertIn('--output-dir "$K35_EXEC_OUTPUT_DIR"', operator_text)

    def test_summarize_results_by_scenario_reports_medians_and_speedups(self):
        summary = k35.summarize_results_by_scenario(
            [
                {
                    "scenario": "frontdoor_cpu_no_spec",
                    "status": "ok",
                    "completion_tokens": 10,
                    "prompt_tokens": 20,
                    "decode_tps": 10.0,
                    "prompt_tps": 100.0,
                    "elapsed_s": 1.0,
                },
                {
                    "scenario": "frontdoor_cpu_no_spec",
                    "status": "ok",
                    "completion_tokens": 10,
                    "prompt_tokens": 20,
                    "decode_tps": 12.0,
                    "prompt_tps": 102.0,
                    "elapsed_s": 1.1,
                },
                {
                    "scenario": "frontdoor_gpu_native_mtp",
                    "status": "ok",
                    "completion_tokens": 10,
                    "prompt_tokens": 20,
                    "decode_tps": 44.0,
                    "prompt_tps": 200.0,
                    "elapsed_s": 0.5,
                    "draft_n": 8,
                    "draft_n_accepted": 6,
                },
            ]
        )
        self.assertEqual(summary["frontdoor_cpu_no_spec"]["decode_tps"]["median"], 11.0)
        self.assertEqual(summary["frontdoor_cpu_no_spec"]["decode_tps"]["mad"], 1.0)
        self.assertEqual(summary["frontdoor_gpu_native_mtp"]["draft_acceptance_rate"], 0.75)
        self.assertEqual(summary["frontdoor_gpu_native_mtp"]["decode_speedup_vs_frontdoor_cpu_no_spec"], 4.0)

    def test_proc_memory_parsers_extract_resident_fields(self):
        status = k35.parse_proc_status(
            "\n".join(
                [
                    "Name:\tllama-server",
                    "VmRSS:\t  123456 kB",
                    "RssAnon:\t   11111 kB",
                    "Cpus_allowed_list:\t0-3",
                    "Ignored:\tvalue",
                ]
            )
        )
        self.assertEqual(status["VmRSS"], "123456 kB")
        self.assertEqual(status["RssAnon"], "11111 kB")
        self.assertEqual(status["Cpus_allowed_list"], "0-3")
        self.assertNotIn("Ignored", status)

        smaps = k35.parse_smaps_rollup(
            "\n".join(
                [
                    "Rss:              2048 kB",
                    "Pss:              1024 kB",
                    "Private_Dirty:     512 kB",
                    "VmFlags: rd wr",
                ]
            )
        )
        self.assertEqual(smaps["Rss_kib"], 2048)
        self.assertEqual(smaps["Pss_kib"], 1024)
        self.assertEqual(smaps["Private_Dirty_kib"], 512)
        self.assertNotIn("VmFlags_kib", smaps)

    def test_collect_process_memory_current_process(self):
        sample = k35.collect_process_memory(os.getpid())
        self.assertEqual(sample["pid"], os.getpid())
        self.assertTrue(sample["status"]["ok"])
        self.assertIn("VmRSS", sample["status"]["fields"])
        self.assertIn("ps", sample)

    def test_collect_rocm_snapshot_preserves_legacy_fields_and_adds_pgpu1_snapshots(self):
        def fake_run_capture(argv, *, timeout=30):
            return {
                "argv": argv,
                "ok": True,
                "returncode": 0,
                "stdout": " ".join(argv),
                "stderr": "",
            }

        with mock.patch.object(k35.shutil, "which", return_value="/usr/bin/rocm-smi"), mock.patch.object(
            k35,
            "run_capture",
            side_effect=fake_run_capture,
        ):
            sample = k35.collect_rocm_snapshot()

        self.assertTrue(sample["ok"])
        self.assertEqual(sample["argv"], ["rocm-smi", "--showpidgpus", "--showmemuse", "--showuse"])
        self.assertEqual(sample["schema"], "epyc.rocm_snapshot.v2")
        self.assertIn("snapshots", sample)
        self.assertIn("clocks", sample["snapshots"])
        self.assertIn("power", sample["snapshots"])
        self.assertIn("temperature", sample["snapshots"])
        self.assertEqual(sample["snapshots"]["clocks"]["argv"], ["rocm-smi", "--showclocks"])

    def test_collect_post_cleanup_sample_records_after_cleanup_rocm(self):
        with mock.patch.object(k35, "run_capture", return_value={"ok": True, "stdout": "", "stderr": ""}), mock.patch.object(
            k35,
            "collect_rocm_snapshot",
            return_value={"ok": True, "stdout": "0% VRAM"},
        ):
            sample = k35.collect_post_cleanup_sample(12345)

        self.assertEqual(sample["phase"], "after_cleanup")
        self.assertEqual(sample["pid"], 12345)
        self.assertEqual(sample["rocm"]["stdout"], "0% VRAM")

    def test_mark_cleanup_failure_overrides_successful_inference_status(self):
        result = {"status": "ok", "scenario": "frontdoor_cpu_no_spec"}
        cleanup = {
            "pid": 12345,
            "returncode": None,
            "dead": False,
            "completed": False,
            "ps_after": {"ok": True, "stdout": "12345 llama-server"},
        }

        k35.mark_cleanup_failure(result, cleanup)

        self.assertEqual(result["status"], "cleanup_failed")
        self.assertEqual(result["inference_status"], "ok")
        self.assertIn("cleanup", result["cleanup_error"])

    def test_cleanup_proved_complete_accepts_ps_absence_returncode(self):
        cleanup = {
            "pid": 12345,
            "returncode": -15,
            "dead": True,
            "completed": True,
            "ps_after": {"ok": False, "returncode": 1, "stdout": ""},
        }

        self.assertTrue(k35.cleanup_proved_complete(cleanup))

    def test_execute_plan_fails_summary_when_cleanup_not_proved(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = k35.parse_args(["--execute", "--output-dir", tmp, "--allow-dirty-host"])
            plan = {"cells": [{"scenario": "frontdoor_cpu_no_spec"}], "pgpu1_protocol_fields": {"policy": "test"}}
            leaked_result = {
                "scenario": "frontdoor_cpu_no_spec",
                "nominal_context": 2048,
                "rep": 1,
                "status": "cleanup_failed",
                "inference_status": "ok",
                "cleanup": {
                    "pid": 12345,
                    "returncode": None,
                    "dead": False,
                    "completed": False,
                    "ps_after": {"ok": True, "stdout": "12345 llama-server"},
                },
            }

            with mock.patch.object(k35, "collect_guard_state", return_value={"process_blockers": []}), mock.patch.object(
                k35,
                "run_cell",
                return_value=leaked_result,
            ), mock.patch.object(k35, "collect_process_blockers", return_value=[]):
                summary = k35.execute_plan(plan, args, Path(tmp))

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["results"][0]["status"], "cleanup_failed")
        self.assertEqual(summary["cleanup_failures"][0]["inference_status"], "ok")


if __name__ == "__main__":
    unittest.main()
