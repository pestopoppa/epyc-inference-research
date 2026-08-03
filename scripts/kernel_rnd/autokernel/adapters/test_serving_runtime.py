#!/usr/bin/env python3
"""Unit tests for serving_runtime.py — the stack-change adapter (§11.6, §13.5).

NO inference, NO benchmark, NO kernel build, NO stack. Nothing here starts,
stops, or signals a service. The only subprocess any test creates is a throwaway
Python script written into that test's own temp directory, used to exercise the
`GuardRunner` seam end-to-end against the real `stack_change_guard.py` OUTPUT
contract — never against the real guard, and never against the real stack.

The suite is organised around what this adapter must refuse:

  * the kernel-freeze path, in every direction it could be reached from — by
    name, by a forbidden action id, by a command string, by a package field, and
    by a candidate that quietly changed the kernel binary;
  * one gate's evidence answering another gate, which is `feedback_stack_change_
    three_gates` in data-contract form;
  * gate 3 verified against the config file or the topology hash rather than the
    live process;
  * tokens/s anywhere in serving scope, and task_rate anywhere else;
  * a fixed-shape benchmark relabelled as serving evidence;
  * a composed estimate assembled by adding local percentages;
  * a cross-engine whole-stack ratio recruited as an objective — while every
    batch regime stays admissible (AK-D36 vs AK-D37).

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/adapters/test_serving_runtime.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/adapters/test_serving_runtime.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE, never by putting this directory on sys.path.
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel.adapters import serving_runtime as SR  # noqa: E402

CAMPAIGN = "ak-serving-sched-20260803"
CANDIDATE = "akc-serving-admission-001"

# A fixture-only protocol id. `measurement/protocols/` contains no serving
# protocol yet — authoring one is AK9-class work and is deliberately NOT done
# here (this module takes the protocol id as a caller parameter and hardcodes
# none).
SERVING_PROTOCOL = "P-AK-SERVING-REPLAY-FIXTURE/v0"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _commit(tag: str) -> str:
    return _sha(tag)[:40]


def _pinned(**overrides) -> SR.PinnedProductionConfig:
    kwargs = dict(
        config_id="stack-prod-20260801",
        config_sha256=_sha("pinned-config"),
        engine="llama-server",
        roles=("architect", "worker"),
        kernel_binary_sha256=_sha("kernel-binary-v8"),
        kernel_linkage_sha256=_sha("kernel-linkage-v8"),
        pinned_at="2026-08-01T09:00:00+00:00",
        pinned_bench_receipt="bench://cpu-pinned/20260801",
        archive_path="/mnt/raid0/llm/kernels/archive/stack-prod-20260801.tar.zst",
        archive_sha256=_sha("pinned-archive"),
    )
    kwargs.update(overrides)
    return SR.PinnedProductionConfig(**kwargs)


def _trace(**overrides) -> SR.ArrivalTrace:
    kwargs = dict(
        trace_id="arrivals-mixed-20260731",
        trace_sha256=_sha("trace"),
        seed=42,
        request_count=4096,
        duration_s=1800.0,
        roles=("architect", "worker"),
    )
    kwargs.update(overrides)
    return SR.ArrivalTrace(**kwargs)


def _workload(**overrides) -> SR.VariableArrivalReplaySpec:
    kwargs = dict(
        spec_id="replay-mixed-16c",
        benchmark_class=SR.BENCHMARK_CLASS,
        trace=_trace(),
        concurrency=16,
        warmup_s=60.0,
        recipe_constructor_id="autokernel.evaluator.recipes/v1",
        recipe_sha256=_sha("recipe"),
    )
    kwargs.update(overrides)
    return SR.VariableArrivalReplaySpec(**kwargs)


def _evidence(**overrides) -> SR.ServingEvidence:
    kwargs = dict(
        campaign_id=CAMPAIGN,
        candidate_id=CANDIDATE,
        workload=_workload(),
        task_rate=SR.TaskRateCell(value=3.24, raw_samples_ref="raw://task-rate",
                                  paired_blocks=12),
        latency=(
            SR.LatencyCell(name="p50", value_ms=820.0, raw_samples_ref="raw://lat"),
            SR.LatencyCell(name="p95", value_ms=2410.0, raw_samples_ref="raw://lat"),
        ),
        slo=(
            SR.SloCell(slo_id="p95_under_3s", target_description="p95 <= 3000 ms",
                       attainment=0.987, window_s=1800.0, raw_samples_ref="raw://slo"),
        ),
        comparison_config_id="stack-prod-20260801",
        comparison_config_sha256=_sha("pinned-config"),
        engine="llama-server",
    )
    kwargs.update(overrides)
    return SR.ServingEvidence(**kwargs)


def _linkage(tree="llama.cpp", status=S.PASS) -> SR.LinkageReceipt:
    return SR.LinkageReceipt(
        verifier=SR.LINKAGE_VERIFIER, status=status, resolved_tree=tree,
        receipt_ref=f"linkage://{tree}",
    )


def _service(service_id="worker", index=0, start="2026-08-03T10:00:00+00:00",
             ready="2026-08-03T10:01:00+00:00", tree="llama.cpp",
             ld=None, linkage=..., state="running", pid=1000):
    if ld is None:
        ld = "/mnt/raid0/llm/llama.cpp/build/bin:/usr/lib"
    if linkage is ...:
        linkage = _linkage(tree) if tree != SR.NO_GGML_TREE else None
    return SR.ServiceStartObservation(
        service_id=service_id, tree=tree, start_index=index, started_at=start,
        ready_at=ready, pid=pid, state=state, ld_library_path=ld, linkage=linkage,
    )


#: The stable production kernel path a serving binary is executed THROUGH. It is
#: the fixture's binary because it is the real one: §11.6's whole point is that
#: serving runs the frozen kernel from the stable path.
_PROD_LANE = "/mnt/raid0/llm/kernels/production/cpu"
_PROD_BINARY = _PROD_LANE + "/llama-server"


def _intended(service_id="worker", **overrides) -> SR.IntendedProcessConfig:
    kwargs = dict(
        service_id=service_id,
        binary_path=_PROD_BINARY,
        binary_sha256=_sha("kernel-binary-v8"),
        flags=("--parallel", "8", "--cont-batching"),
        cpu_affinity=(0, 1, 2, 3),
        config_sha256=_sha("pinned-config"),
        config_recorded_at="2026-08-03T09:00:00+00:00",
        flags_are_exhaustive=True,
    )
    kwargs.update(overrides)
    return SR.IntendedProcessConfig(**kwargs)


def _live(service_id="worker", **overrides) -> SR.LiveProcessFact:
    kwargs = dict(
        service_id=service_id,
        pid=4242,
        observation_source="proc_cmdline",
        binary_path=_PROD_BINARY,
        binary_sha256=_sha("kernel-binary-v8"),
        argv=(_PROD_BINARY, "--parallel", "8", "--cont-batching"),
        cpu_affinity=(0, 1, 2, 3),
        started_at="2026-08-03T10:00:00+00:00",
    )
    kwargs.update(overrides)
    return SR.LiveProcessFact(**kwargs)


def _started(*facts, start="2026-08-03T09:59:00+00:00",
             ready="2026-08-03T10:05:00+00:00"):
    """Gate-2 observations that TIE to the given live facts.

    Gate 3 is now decided about the processes gate 2 measured, so every gate-3
    fixture states which gate-2 observation it belongs to. The default window
    brackets `_live()`'s start instant, which is what a stack that came up and
    then stayed up looks like.
    """
    return tuple(
        _service(service_id=fact.service_id, index=i, pid=fact.pid,
                 start=start, ready=ready)
        for i, fact in enumerate(facts)
    )


#: The one process every hand-built gate fixture is about. Gate 2 declares it as
#: its subject and gate 3 declares it as the gate-2 subject it was decided
#: against, which is what a real pair of gate functions emits — the release
#: verdict now requires that binding to exist and to agree.
_FIXTURE_SUBJECT = ("worker#4242",)


def _gate(gate, status=S.PASS, reasons=(), kind=None, ref="ref",
          subjects=None, tied_to=None):
    kinds = {SR.GATE_1: "stack_change_guard_result",
             SR.GATE_2: "service_start_observation",
             SR.GATE_3: "live_process_observation"}
    default_subjects = () if gate == SR.GATE_1 else _FIXTURE_SUBJECT
    default_tied = _FIXTURE_SUBJECT if gate == SR.GATE_3 else ()
    return SR.GateOutcome(gate=gate, status=status, evidence_kind=kind or kinds[gate],
                          evidence_ref=ref, reasons=reasons,
                          subjects=default_subjects if subjects is None else subjects,
                          tied_to=default_tied if tied_to is None else tied_to)


def _all_gates_pass() -> SR.ThreeGateResult:
    return SR.evaluate_three_gates(
        pipeline_green=_gate(SR.GATE_1),
        stack_starts=_gate(SR.GATE_2),
        live_equals_config=_gate(SR.GATE_3),
    )


def _rollback(**overrides) -> SR.RollbackPlan:
    kwargs = dict(
        incumbent_config_id="stack-prod-20260801",
        incumbent_config_sha256=_sha("pinned-config"),
        archive_path="/mnt/raid0/llm/config-archive/stack-prod-20260801.tar.zst",
        archive_sha256=_sha("pinned-archive"),
        archive_verified=True,
        restore_command=("orchestrator_stack.py apply --config "
                         "/mnt/raid0/llm/config-archive/stack-prod-20260801.yaml"),
    )
    kwargs.update(overrides)
    return SR.RollbackPlan(**kwargs)


def _commands():
    return (
        {"command": "orchestrator_stack.py reload orchestrator",
         "validated": True, "validation_receipt": "dryrun://reload/1"},
    )


def _waiver(campaign_id=CAMPAIGN, **overrides) -> dict:
    record = {
        "schema": S.SCHEMA_OPERATOR_WAIVER,
        "waiver_id": "WAIVE-SERVING-TAIL-1",
        "campaign_id": campaign_id,
        "decision": "accept with scope",
        "protocol": SERVING_PROTOCOL,
        "protocol_changed": False,
        "candidate_head": _commit("candidate"),
        "production_head": _commit("production"),
        "scope": {
            "excluded_models": ["gemma4-26B-A4B"],
            "excluded_pairs": [["gemma4-26B-A4B", "p99"]],
            "remaining_matched_pairs": 11,
        },
        "reason": "p99 tail on one role is dominated by a known loader stall",
        "consequences": ["no p99 non-regression claim for gemma4-26B-A4B"],
        "authorized_by": "operator",
        "expiry": {"expires_at": "2026-11-01T00:00:00+00:00",
                   "reopen_predicate": "loader stall fixed"},
        "created_at": "2026-08-03T08:00:00+00:00",
    }
    record.update(overrides)
    return record


def _package(**overrides) -> SR.StackChangePackage:
    kwargs = dict(
        package_id="aks-serving-20260803-1",
        campaign_id=CAMPAIGN,
        gates=_all_gates_pass(),
        evidence=_evidence(),
        pinned=_pinned(),
        rollback=_rollback(),
        blast_radius=SR.classify_blast_radius(diff_lines=120, files_touched=4,
                                              touches_shared_core=False),
        operator_command_sequence=_commands(),
        candidate_binary_sha256=_sha("kernel-binary-v8"),
        candidate_linkage_sha256=_sha("kernel-linkage-v8"),
        created_at="2026-08-03T12:00:00+00:00",
    )
    kwargs.update(overrides)
    return SR.assemble_stack_change_package(**kwargs)


# =============================================================================


class TestIdentity(unittest.TestCase):
    """The adapter's vocabulary must agree with the shared data contracts."""

    def test_backend_and_lane_are_the_shared_vocabulary(self):
        self.assertIn(SR.BACKEND, S.BACKENDS)
        self.assertIn(SR.RESOURCE_LANE, S.RESOURCE_LANES)
        self.assertEqual(SR.METRIC, "task_rate")
        self.assertEqual(SR.RELEASE_PATH, "stack_change")

    def test_serving_runtime_has_no_source_tree(self):
        # §1.5: four binaries, three trees, and serving is in none of them. Its
        # worktree ownership is a mapping, which is why it cannot be frozen.
        self.assertNotIn(SR.BACKEND, S.SOURCE_TREE_BY_BACKEND)

    def test_tree_roots_cover_every_source_tree(self):
        self.assertEqual(set(SR.TREE_ROOTS), set(S.SOURCE_TREES))
        self.assertEqual(SR.SERVICE_TREES,
                         frozenset(S.SOURCE_TREES) | {SR.NO_GGML_TREE})

    def test_adapter_id_is_versioned(self):
        # A record naming a mutable adapter id cannot be replayed.
        self.assertRegex(SR.ADAPTER_ID, r"/v\d+$")

    def test_linkage_verifier_lives_in_the_research_repo(self):
        # §10.2: CLAUDE.md cites it unqualified; it is in epyc-inference-research.
        self.assertTrue(SR.LINKAGE_VERIFIER.startswith(
            "/mnt/raid0/llm/epyc-inference-research/"))


class TestKernelFreezeRefusal(unittest.TestCase):
    """The cardinal rule: this adapter never travels the kernel-freeze path."""

    def test_refuse_kernel_freeze_always_raises(self):
        for request in ("freeze", "seal_release_candidate", "cutover", ""):
            with self.assertRaises(SR.KernelFreezePathRefused):
                SR.refuse_kernel_freeze(request or "unnamed")

    def test_release_path_is_stack_change_and_only_for_this_backend(self):
        self.assertEqual(SR.release_path_for("serving_runtime"), "stack_change")
        for other in ("llama_cpu", "llama_gpu", "whisper_stt", "qwentts_tts"):
            with self.assertRaises(SR.ServingAdapterError):
                SR.release_path_for(other)

    def test_scan_catches_every_human_only_action_id(self):
        for action in sorted(SR.FORBIDDEN_PRODUCTION_ACTIONS):
            check = SR.scan_for_kernel_freeze_actions({"steps": [{"action": action}]})
            self.assertEqual(check.outcome, S.FAIL, action)
            self.assertTrue(any(action in r for r in check.reasons))

    def test_scan_catches_production_write_command_strings(self):
        cases = [
            "git checkout production-consolidated-v9",
            "ln -sfn /mnt/raid0/llm/llama.cpp/build/bin /mnt/raid0/llm/kernels/production/cpu",
            "vim orchestration/instrument_eras.yaml",
            "python3 apply.py orchestration/autopilot_baseline.yaml",
            "edit human_only_paths.sha256",
            "bash freeze_v9_production_20260901.sh",
            "sudo reboot",
        ]
        for command in cases:
            check = SR.scan_for_kernel_freeze_actions({"cmd": command})
            self.assertEqual(check.outcome, S.FAIL, command)

    def test_a_declared_action_is_caught_in_a_LIST_not_only_as_a_scalar(self):
        # Regression, red-team 2026-08-03. `_ACTION_KEYS` contains the PLURAL
        # `actions` because a plan's natural shape is a list, but the scan only
        # tested the value when it happened to be a bare string. Every declared
        # human-only action inside a list therefore read as "no forbidden action
        # declared" — and the package body is scanned with
        # match_command_strings=False, so the command patterns did not catch it
        # either. This was a clean route to a package declaring a symlink move.
        for action in sorted(SR.FORBIDDEN_PRODUCTION_ACTIONS):
            check = SR.scan_for_kernel_freeze_actions(
                {"steps": [{"actions": ["reload_service", action]}]},
                match_command_strings=False)
            self.assertEqual(check.outcome, S.FAIL, action)
            self.assertTrue(any(action in r for r in check.reasons), action)

    def test_a_declared_action_is_caught_whatever_its_case_or_padding(self):
        for spelling in ("Freeze", "FREEZE", "  freeze  ", "Write_Era_Registry_Row"):
            check = SR.scan_for_kernel_freeze_actions({"action": spelling},
                                                      match_command_strings=False)
            self.assertEqual(check.outcome, S.FAIL, spelling)

    def test_a_package_cannot_declare_a_production_action_in_a_list(self):
        with self.assertRaises(SR.KernelFreezePathRefused):
            _package(operator_command_sequence=(
                {"command": "orchestrator_stack.py reload orchestrator",
                 "validated": True, "validation_receipt": "dryrun://1",
                 "actions": ["move_stable_kernel_symlink"]},))

    def test_scan_passes_an_ordinary_stack_change_plan(self):
        plan = {"steps": [{"action": "reload_service",
                           "cmd": "orchestrator_stack.py reload orchestrator"}]}
        self.assertEqual(SR.scan_for_kernel_freeze_actions(plan).outcome, S.PASS)

    def test_kernel_artifact_change_is_misrouted_not_recorded(self):
        pinned = _pinned()
        moved = SR.check_no_kernel_artifact_change(
            pinned=pinned, candidate_binary_sha256=_sha("kernel-binary-v9"),
            candidate_linkage_sha256=pinned.kernel_linkage_sha256)
        self.assertEqual(moved.outcome, S.FAIL)

        same = SR.check_no_kernel_artifact_change(
            pinned=pinned, candidate_binary_sha256=pinned.kernel_binary_sha256,
            candidate_linkage_sha256=pinned.kernel_linkage_sha256)
        self.assertEqual(same.outcome, S.PASS)

        unknown = SR.check_no_kernel_artifact_change(
            pinned=pinned, candidate_binary_sha256=None,
            candidate_linkage_sha256=pinned.kernel_linkage_sha256)
        self.assertEqual(unknown.outcome, S.COULD_NOT_CHECK)

    def test_assembly_refuses_a_candidate_whose_kernel_moved(self):
        with self.assertRaises(SR.KernelChangeMisrouted):
            _package(candidate_binary_sha256=_sha("kernel-binary-v9"))

    def test_assembly_refuses_an_unknown_kernel_identity(self):
        # An unknown identity is not an unchanged one.
        with self.assertRaises(SR.ServingAdapterError) as ctx:
            _package(candidate_linkage_sha256="not-a-digest")
        self.assertNotIsInstance(ctx.exception, SR.KernelChangeMisrouted)

    def test_assembly_refuses_a_freeze_command_in_the_operator_sequence(self):
        commands = ({"command": "bash freeze_v9_production_20260901.sh",
                     "validated": True, "validation_receipt": "dryrun://freeze"},)
        with self.assertRaises(SR.KernelFreezePathRefused):
            _package(operator_command_sequence=commands)

    def test_a_failing_gate_may_quote_a_production_path_in_its_reason(self):
        """Diagnostic prose is not an action.

        The guard must not become a reason the adapter cannot REPORT a failure:
        gate 3's natural failure text quotes the production binary path, and a
        whole-record command-pattern scan would refuse the package outright
        instead of handing over a FAIL.
        """
        gates = SR.evaluate_three_gates(
            pipeline_green=_gate(SR.GATE_1),
            stack_starts=_gate(SR.GATE_2),
            live_equals_config=_gate(
                SR.GATE_3, status=S.FAIL,
                reasons=("service 'worker' runs "
                         "'/mnt/raid0/llm/kernels/production/cpu/llama-server', "
                         "configured '/mnt/raid0/llm/candidate/llama-server'",)),
        )
        package = _package(gates=gates)
        self.assertEqual(package.verdict, "FAIL")
        self.assertIn("kernels/production",
                      package.to_record()["gates"]["live_equals_config"]["reasons"][0])

    def test_a_declared_action_is_still_caught_anywhere_in_the_package(self):
        check = SR.scan_for_kernel_freeze_actions(
            {"deep": {"nested": {"action": "write_era_registry_row"}}},
            match_command_strings=False)
        self.assertEqual(check.outcome, S.FAIL)

    def test_rollback_refuses_a_production_restore_command(self):
        with self.assertRaises(SR.KernelFreezePathRefused):
            _rollback(restore_command="git checkout production-consolidated-v8")

    def test_shared_core_diff_is_misrouted_not_merely_marked(self):
        with self.assertRaises(SR.KernelChangeMisrouted):
            SR.classify_blast_radius(diff_lines=3, files_touched=1,
                                     touches_shared_core=True)

    def test_a_stack_change_package_is_not_a_kernel_release_package(self):
        # The two must not be interchangeable. A kernel release package requires
        # a source_tree from SOURCE_TREES, and serving has none.
        record = _package().to_record()
        record_as_release = dict(record)
        record_as_release["schema"] = S.SCHEMA_RELEASE_PACKAGE
        violations = S.validate_release_package(record_as_release)
        self.assertTrue(violations)
        self.assertTrue(any("source_tree" in v for v in violations))


class TestStableKernelPathExecutesVsMutates(unittest.TestCase):
    """AK8 carried-forward: the `kernels/production` substring match forbade the
    adapter's own idiom.

    `/mnt/raid0/llm/kernels/production/<lane>` is the symlink an operator repoints
    at freeze AND the path every serving launcher executes its binary through. A
    substring match cannot tell those apart, so it refused both, and a §11.6
    package could not state the launch command it exists to hand over
    (`feedback_guard_must_not_forbid_its_own_idiom`).

    Both directions are proved here: the launch command is admitted, and every
    shape of repoint, replace, delete and redirect is still refused.
    """

    # -- MUTATES: still refused, in every shape ----------------------------

    def test_a_symlink_repoint_is_still_refused(self):
        command = ("ln -sfn /mnt/raid0/llm/llama.cpp/build/bin " + _PROD_LANE)
        check = SR.classify_stable_kernel_path_use(command)
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        self.assertTrue(any("ITSELF" in r for r in check.reasons), check.reasons)
        self.assertEqual(
            SR.scan_for_kernel_freeze_actions({"cmd": command}).outcome, S.FAIL)

    def test_every_mutating_shape_is_refused(self):
        cases = {
            "repoint": "ln -sfn /mnt/raid0/llm/llama.cpp-v9 " + _PROD_LANE,
            "remove the link": "rm -f " + _PROD_LANE,
            "rename over the link": "mv /mnt/raid0/llm/staging " + _PROD_LANE,
            "quoted link": 'mv /mnt/raid0/llm/staging "' + _PROD_LANE + '"',
            "overwrite the binary": "cp /tmp/llama-server " + _PROD_BINARY,
            "delete the binary": "rm " + _PROD_BINARY,
            "chmod the binary": "chmod 0755 " + _PROD_BINARY,
            "install into the tree": "install -m 0755 x " + _PROD_LANE + "/bin/y",
            "redirect into the tree": "echo hi > " + _PROD_LANE + "/bin/marker",
            "make with DESTDIR": "DESTDIR=" + _PROD_LANE + "/bin make install",
            "second segment": _PROD_BINARY + " --port 8080 && ln -sfn a " + _PROD_LANE,
            "piped into the tree": "cat x | tee " + _PROD_LANE + "/bin/marker",
            "shell substitution": "rm $(readlink " + _PROD_LANE + ")",
        }
        for label, command in cases.items():
            with self.subTest(label):
                self.assertEqual(SR.classify_stable_kernel_path_use(command).outcome,
                                 S.FAIL, command)
                self.assertEqual(
                    SR.scan_for_kernel_freeze_actions({"cmd": command}).outcome,
                    S.FAIL, command)

    def test_a_launcher_prefix_cannot_smuggle_a_mutating_verb_into_command_position(self):
        """The passthrough walks to the program a launcher launches, not past it.

        `taskset -c 0-95 rm <path>` must report `rm`, not the path: a launcher
        prefix that stepped over any bare word would make every mutation legal by
        prefixing it with `taskset`.
        """
        command = "taskset -c 0-95 rm " + _PROD_BINARY
        check = SR.classify_stable_kernel_path_use(command)
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        self.assertTrue(any("'rm'" in r for r in check.reasons), check.reasons)

    def test_a_non_loader_assignment_prefix_is_refused(self):
        # Only the three dynamic-loader search paths may name the tree; every
        # other NAME=value is a build/install destination.
        for name in ("DESTDIR", "PREFIX", "OUT", "TARGET"):
            command = name + "=" + _PROD_LANE + "/bin cmake --install ."
            self.assertEqual(SR.classify_stable_kernel_path_use(command).outcome,
                             S.FAIL, command)

    def test_a_read_only_use_by_another_program_is_refused_conservatively(self):
        # The classifier is conservative by design: a false positive costs a human
        # one look, a false negative costs a production write. `ls` is refused
        # because "the command word is what executes" is the whole rule.
        self.assertEqual(
            SR.classify_stable_kernel_path_use("ls -l " + _PROD_LANE + "/bin").outcome,
            S.FAIL)

    # -- EXECUTES: the adapter's own idiom, admitted -----------------------

    def test_the_normal_launch_command_is_admitted(self):
        command = (_PROD_BINARY + " -m /mnt/raid0/llm/models/qwen36-q8.gguf "
                   "-t 96 -fa 1 --port 8080")
        check = SR.classify_stable_kernel_path_use(command)
        self.assertEqual(check.outcome, S.PASS, check.reasons)
        self.assertEqual(
            SR.scan_for_kernel_freeze_actions({"cmd": command}).outcome, S.PASS)

    def test_every_real_launcher_shape_is_admitted(self):
        cases = {
            "bare": _PROD_BINARY + " --port 8080",
            "taskset": "taskset -c 0-95 " + _PROD_BINARY + " -t 96 -fa 1",
            "taskset SMT siblings": "taskset -c 184-191 " + _PROD_BINARY,
            "numactl": "numactl --interleave=all " + _PROD_BINARY + " -t 96",
            "numactl short": "numactl -i all " + _PROD_BINARY,
            # CLAUDE.md's speech-kernel freeze: every launcher must set its own
            # LD_LIBRARY_PATH. A guard that refused that refused the mandated form.
            "own LD_LIBRARY_PATH": ("LD_LIBRARY_PATH=" + _PROD_LANE + "/lib "
                                    + _PROD_BINARY + " --port 8080"),
            "env + taskset + LD_LIBRARY_PATH": (
                "env LD_LIBRARY_PATH=" + _PROD_LANE + "/lib:/usr/lib "
                "taskset -c 184-191 " + _PROD_BINARY + " -fa 1"),
            "nohup setsid": "nohup setsid " + _PROD_BINARY + " --port 8080",
            "its own tree as an operand": (
                _PROD_BINARY + " --lora " + _PROD_LANE + "/share/adapter.gguf"),
        }
        for label, command in cases.items():
            with self.subTest(label):
                self.assertEqual(SR.classify_stable_kernel_path_use(command).outcome,
                                 S.PASS, command)
                self.assertEqual(
                    SR.scan_for_kernel_freeze_actions({"cmd": command}).outcome,
                    S.PASS, command)

    def test_a_package_can_report_its_own_serving_launch_command(self):
        """The defect, end to end: a realistic §11.6 package assembles.

        Before the distinction existed this raised `KernelFreezePathRefused` — the
        package could not name the binary it exists to describe.
        """
        commands = (
            {"command": "orchestrator_stack.py reload orchestrator",
             "validated": True, "validation_receipt": "dryrun://reload/1"},
            {"command": ("taskset -c 0-95 " + _PROD_BINARY
                         + " -m /mnt/raid0/llm/models/qwen36-q8.gguf -t 96 -fa 1"),
             "target_paths": [_PROD_BINARY],
             "validated": True, "validation_receipt": "dryrun://launch/1"},
        )
        package = _package(operator_command_sequence=commands)
        self.assertIn(_PROD_BINARY,
                      package.to_record()["operator_command_sequence"][1]["command"])

    def test_a_package_still_cannot_repoint_the_stable_path(self):
        commands = (
            {"command": "ln -sfn /mnt/raid0/llm/llama.cpp-v9 " + _PROD_LANE,
             "validated": True, "validation_receipt": "dryrun://repoint"},
        )
        with self.assertRaises(SR.KernelFreezePathRefused):
            _package(operator_command_sequence=commands)

    def test_a_rollback_may_launch_the_incumbent_but_not_repoint_it(self):
        allowed = _rollback(restore_command=_PROD_BINARY + " --port 8080")
        self.assertIn(_PROD_BINARY, allowed.restore_command)
        with self.assertRaises(SR.KernelFreezePathRefused):
            _rollback(restore_command="ln -sfn /mnt/raid0/llm/llama.cpp-v9 "
                                      + _PROD_LANE)

    # -- the classifier cannot be passed by deleting what it inspects ------

    def test_a_command_that_never_names_the_path_is_could_not_check_not_pass(self):
        check = SR.classify_stable_kernel_path_use(
            "orchestrator_stack.py reload orchestrator")
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(any("no use of the stable production kernel path"
                            in r for r in check.reasons), check.reasons)

    def test_a_non_string_is_could_not_check_not_pass(self):
        for value in (None, 42, [_PROD_BINARY], {"cmd": _PROD_BINARY}):
            self.assertEqual(SR.classify_stable_kernel_path_use(value).outcome,
                             S.COULD_NOT_CHECK, value)

    def test_diagnostic_prose_is_still_not_scanned_as_a_command(self):
        # The whole-body scan runs with match_command_strings=False, so relaxing
        # the classifier changed nothing about a gate reason quoting the path.
        check = SR.scan_for_kernel_freeze_actions(
            {"reason": "service 'worker' runs " + _PROD_BINARY},
            match_command_strings=False)
        self.assertEqual(check.outcome, S.PASS)


class TestGateFramework(unittest.TestCase):
    """Three gates, distinct, in order, none implying the next (§11.6)."""

    def test_gate_refuses_another_gates_evidence(self):
        with self.assertRaises(SR.GateEvidenceMisuse) as ctx:
            SR.GateOutcome(gate=SR.GATE_2, status=S.PASS,
                           evidence_kind="stack_change_guard_result",
                           evidence_ref="guard")
        self.assertIn("pipeline_green", str(ctx.exception))

    def test_gate3_refuses_the_config_file_and_the_topology_hash(self):
        for kind in ("config_file", "topology_hash", "intended_config", "assumed"):
            with self.assertRaises(SR.GateEvidenceMisuse) as ctx:
                SR.GateOutcome(gate=SR.GATE_3, status=S.PASS, evidence_kind=kind,
                               evidence_ref="x")
            self.assertIn(kind, str(ctx.exception))

    def test_gate1_refuses_autopilot_state_as_evidence(self):
        with self.assertRaises(SR.GateEvidenceMisuse):
            SR.GateOutcome(gate=SR.GATE_1, status=S.PASS,
                           evidence_kind="autopilot_state", evidence_ref="x")

    def test_pass_with_reasons_and_nonpass_without_reasons_are_refused(self):
        with self.assertRaises(ValueError):
            _gate(SR.GATE_1, reasons=("something",))
        with self.assertRaises(ValueError):
            _gate(SR.GATE_1, status=S.FAIL, reasons=())

    def test_all_three_pass_is_the_only_release_eligible_state(self):
        result = _all_gates_pass()
        self.assertEqual(result.status, S.PASS)
        self.assertTrue(result.released)
        self.assertIsNone(result.blocked_at)
        result.require_release_eligible()

    def test_unattempted_gate_is_could_not_check_never_pass(self):
        result = SR.evaluate_three_gates(pipeline_green=_gate(SR.GATE_1))
        self.assertEqual(result.status, S.COULD_NOT_CHECK)
        self.assertEqual(result.blocked_at, SR.GATE_2)
        self.assertFalse(result.released)
        with self.assertRaises(SR.ServingAdapterError):
            result.require_release_eligible()

    def test_a_failed_gate_blocks_and_names_itself(self):
        result = SR.evaluate_three_gates(
            pipeline_green=_gate(SR.GATE_1),
            stack_starts=_gate(SR.GATE_2, status=S.FAIL, reasons=("did not start",)),
        )
        self.assertEqual(result.status, S.FAIL)
        self.assertEqual(result.blocked_at, SR.GATE_2)
        self.assertTrue(any("did not start" in r for r in result.reasons))

    def test_a_later_gate_may_not_be_supplied_past_a_failed_one(self):
        with self.assertRaises(SR.GateOrderViolation):
            SR.evaluate_three_gates(
                pipeline_green=_gate(SR.GATE_1, status=S.FAIL, reasons=("red",)),
                live_equals_config=_gate(SR.GATE_3),
            )

    def test_a_gate_outcome_in_the_wrong_position_is_refused(self):
        with self.assertRaises(SR.GateOrderViolation):
            SR.evaluate_three_gates(pipeline_green=_gate(SR.GATE_1),
                                    stack_starts=_gate(SR.GATE_3))

    def test_verdict_cannot_be_attached_to_gates_that_do_not_support_it(self):
        with self.assertRaises(SR.GateVerdictTampering):
            SR.ThreeGateResult(
                pipeline_green=_gate(SR.GATE_1, status=S.FAIL, reasons=("red",)),
                stack_starts=None, live_equals_config=None,
                status=S.PASS, blocked_at=None,
            )


class TestGate1PipelineGreen(unittest.TestCase):
    """Gate 1 models `stack_change_guard.py`'s CLI contract."""

    def test_invocation_is_constructed_not_hand_typed(self):
        inv = SR.build_guard_invocation()
        self.assertIn("--strict", inv.argv)
        self.assertIn("--priors", inv.argv)
        self.assertTrue(inv.argv[1].endswith("stack_change_guard.py"))

    def test_invocation_flag_and_argv_must_agree(self):
        with self.assertRaises(ValueError):
            SR.GuardInvocation(argv=("python3", "guard.py"), cwd="/tmp",
                               priors_path="/tmp/p.yaml", strict=True)

    def test_summary_only_is_refused_because_a_count_is_not_an_enumeration(self):
        with self.assertRaises(ValueError):
            SR.GuardInvocation(argv=("python3", "g.py", "--surface-summary-only"),
                               cwd="/tmp", priors_path="/tmp/p.yaml", strict=False)

    def test_parses_ok_warn_and_fail(self):
        inv = SR.build_guard_invocation()
        ok = SR.parse_guard_result(inv, 0, "OK: /x/stack_priors.yaml\n")
        self.assertEqual(ok.header, "OK")
        self.assertTrue(ok.parsed)
        self.assertEqual(ok.defects, ())

        warn = SR.parse_guard_result(
            inv, 0, "WARN: 2 stack-prior warning(s)\n  - a\n  - b\n")
        self.assertEqual(warn.warnings, ("a", "b"))

        fail = SR.parse_guard_result(
            inv, 1, "FAIL: 1 stack-prior error(s)\n  - boom\n")
        self.assertEqual(fail.errors, ("boom",))

    def test_exit_code_body_disagreement_is_a_defect_not_a_preference(self):
        inv = SR.build_guard_invocation()
        result = SR.parse_guard_result(inv, 0, "FAIL: 1 stack-prior error(s)\n  - x\n")
        self.assertTrue(result.defects)
        outcome = SR.gate_pipeline_green(result)
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(outcome.defect)

    def test_count_mismatch_is_a_defect(self):
        inv = SR.build_guard_invocation()
        result = SR.parse_guard_result(inv, 0, "WARN: 3 stack-prior warning(s)\n  - a\n")
        self.assertTrue(result.defects)

    def test_unparseable_output_is_could_not_check(self):
        inv = SR.build_guard_invocation()
        result = SR.parse_guard_result(inv, 0, "everything is fine, trust me\n")
        self.assertFalse(result.parsed)
        outcome = SR.gate_pipeline_green(result)
        self.assertEqual(outcome.status, S.COULD_NOT_CHECK)

    def test_exit_zero_with_an_unaccepted_warning_fails(self):
        # The guard exits 0 on warnings. Taking the exit code as green is exactly
        # the stale-consumer-surface hole gate 1 exists to close.
        inv = SR.build_guard_invocation()
        result = SR.parse_guard_result(
            inv, 0, "WARN: 1 stack-prior warning(s)\n  - hardcoded role in x.py\n")
        outcome = SR.gate_pipeline_green(result)
        self.assertEqual(outcome.status, S.FAIL)

    def test_pre_declared_warning_passes_and_is_recorded(self):
        inv = SR.build_guard_invocation()
        result = SR.parse_guard_result(
            inv, 0, "WARN: 1 stack-prior warning(s)\n  - hardcoded role in x.py\n")
        outcome = SR.gate_pipeline_green(
            result, accepted_warnings=("hardcoded role in x.py",))
        self.assertEqual(outcome.status, S.PASS)
        self.assertTrue(any("accepted guard warning" in n for n in outcome.notes))

    def test_stale_acceptance_is_surfaced_not_dropped(self):
        inv = SR.build_guard_invocation()
        result = SR.parse_guard_result(inv, 0, "OK: /x/stack_priors.yaml\n")
        outcome = SR.gate_pipeline_green(result, accepted_warnings=("gone",))
        self.assertEqual(outcome.status, S.PASS)
        self.assertTrue(any("stale acceptance" in n for n in outcome.notes))

    def test_non_strict_run_cannot_satisfy_a_release_gate(self):
        inv = SR.build_guard_invocation(strict=False)
        result = SR.parse_guard_result(inv, 0, "OK: /x/stack_priors.yaml\n")
        outcome = SR.gate_pipeline_green(result)
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("--strict" in r for r in outcome.reasons))

    def test_errors_fail(self):
        inv = SR.build_guard_invocation()
        result = SR.parse_guard_result(
            inv, 1, "FAIL: 1 stack-prior error(s)\n  - retired role live\n")
        self.assertEqual(SR.gate_pipeline_green(result).status, S.FAIL)

    def test_a_guard_run_with_its_scan_switched_off_cannot_pass_gate_1(self):
        # Regression, red-team 2026-08-03. Gate 1 checked that `--strict` was
        # PRESENT but never that a scope-narrowing flag was ABSENT. The guard's
        # `--skip-hardcoded-surface-scan` and `--allow-production-blocker-waivers`
        # leave the output shape intact — still `OK:`, still exit 0 — so the gate
        # was passable by deleting the thing it inspects rather than by
        # satisfying it. `build_guard_invocation()` cannot emit either flag, but
        # `GuardInvocation` is exported and a recorded run is a caller's argv.
        self.assertTrue(SR.GUARD_SCOPE_WEAKENING_FLAGS)
        for flag in SR.GUARD_SCOPE_WEAKENING_FLAGS:
            invocation = SR.GuardInvocation(
                argv=("python3", str(SR.GUARD_SCRIPT), "--priors", "/p.yaml",
                      "--strict", flag),
                cwd="/tmp", priors_path="/p.yaml", strict=True)
            outcome = SR.gate_pipeline_green(
                SR.parse_guard_result(invocation, 0, "OK: /p.yaml\n"))
            self.assertEqual(outcome.status, S.FAIL, flag)
            self.assertTrue(any(flag in r for r in outcome.reasons), flag)

    def test_the_codified_invocation_still_passes(self):
        # The guard above must not forbid its own idiom: the invocation this
        # module constructs carries none of those flags and still passes.
        invocation = SR.build_guard_invocation(priors_path="/p.yaml")
        self.assertEqual(
            SR.gate_pipeline_green(
                SR.parse_guard_result(invocation, 0, "OK: /p.yaml\n")).status, S.PASS)

    def test_the_weakening_flags_are_flags_the_real_guard_accepts(self):
        # A flag list that has drifted from the guard's CLI protects nothing.
        if not SR.GUARD_SCRIPT.exists():
            self.skipTest(f"{SR.GUARD_SCRIPT} is not present on this host")
        source = SR.GUARD_SCRIPT.read_text(encoding="utf-8")
        for flag in SR.GUARD_SCOPE_WEAKENING_FLAGS:
            self.assertIn(f'"{flag}"', source, flag)

    def test_run_guard_without_a_runner_is_a_refusal_not_a_pass(self):
        with self.assertRaises(SR.GuardRunnerNotWired):
            SR.run_guard(SR.build_guard_invocation(), None)

    def test_run_guard_rejects_a_malformed_runner_result(self):
        with self.assertRaises(SR.ServingAdapterError):
            SR.run_guard(SR.build_guard_invocation(), lambda inv: "OK")

    def test_seam_end_to_end_against_a_subprocess_this_test_creates(self):
        """The only process any test in this file starts: a temp stub guard.

        It emulates the real guard's OUTPUT contract, never the real guard, and
        it touches no repository. This proves the seam composes with a genuine
        subprocess runner without this module ever spawning one.
        """
        tmp = tempfile.mkdtemp(prefix="ak-serving-guard-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        stub = os.path.join(tmp, "stub_guard.py")
        with open(stub, "w", encoding="utf-8") as handle:
            handle.write(
                "import sys\n"
                "print('WARN: 1 stack-prior warning(s)')\n"
                "print('  - hardcoded role in x.py')\n"
                "sys.exit(0)\n"
            )

        def runner(invocation):
            completed = subprocess.run(list(invocation.argv), capture_output=True,
                                       text=True, timeout=60, check=False)
            return completed.returncode, completed.stdout, completed.stderr

        inv = SR.build_guard_invocation(script=stub, repo_root=tmp,
                                        priors_path=os.path.join(tmp, "p.yaml"))
        result = SR.run_guard(inv, runner)
        self.assertEqual(result.warnings, ("hardcoded role in x.py",))
        self.assertEqual(
            SR.gate_pipeline_green(result,
                                   accepted_warnings=("hardcoded role in x.py",)).status,
            S.PASS,
        )
        self.assertEqual(SR.gate_pipeline_green(result).status, S.FAIL)

    def test_the_real_guard_still_prints_what_this_gate_parses(self):
        """Read-only contract check against the actual orchestrator script.

        Gate 1 is built on that script's output format. If it changes, this gate
        silently mis-parses, so the coupling is asserted rather than assumed. The
        file is only READ; that repository is off limits for writes.
        """
        if not SR.GUARD_SCRIPT.exists():
            self.skipTest(f"{SR.GUARD_SCRIPT} is not present on this host")
        source = SR.GUARD_SCRIPT.read_text(encoding="utf-8")
        for fragment in ('stack-prior error(s)', 'stack-prior warning(s)',
                         'f"OK: {args.priors}"'):
            self.assertIn(fragment, source,
                          f"guard output contract moved: {fragment!r} not found")


class TestGate2StackStarts(unittest.TestCase):
    """Gate 2: every affected service comes up, sequentially, linked to its own tree."""

    def test_happy_path(self):
        obs = [_service("worker", 0),
               _service("api", 1, start="2026-08-03T10:02:00+00:00",
                        ready="2026-08-03T10:02:30+00:00", tree=SR.NO_GGML_TREE,
                        ld="/usr/lib", pid=1001)]
        outcome = SR.gate_stack_starts(obs, affected_services=("worker", "api"))
        self.assertEqual(outcome.status, S.PASS, outcome.reasons)

    def test_a_missing_affected_service_fails(self):
        outcome = SR.gate_stack_starts([_service("worker", 0)],
                                       affected_services=("worker", "api"))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("api" in r for r in outcome.reasons))

    def test_an_undeclared_observed_service_is_a_scope_disagreement(self):
        obs = [_service("worker", 0),
               _service("ghost", 1, start="2026-08-03T10:02:00+00:00",
                        ready="2026-08-03T10:03:00+00:00", pid=1002)]
        outcome = SR.gate_stack_starts(obs, affected_services=("worker",))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("ghost" in r for r in outcome.reasons))

    def test_a_service_that_is_not_running_fails(self):
        outcome = SR.gate_stack_starts([_service("worker", 0, state="exited")],
                                       affected_services=("worker",))
        self.assertEqual(outcome.status, S.FAIL)

    def test_overlapping_starts_are_not_sequential(self):
        obs = [_service("worker", 0, start="2026-08-03T10:00:00+00:00",
                        ready="2026-08-03T10:05:00+00:00"),
               _service("api", 1, start="2026-08-03T10:01:00+00:00",
                        ready="2026-08-03T10:06:00+00:00", tree=SR.NO_GGML_TREE,
                        ld="", pid=1003)]
        outcome = SR.gate_stack_starts(obs, affected_services=("worker", "api"))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("sequential" in r for r in outcome.reasons))

    def test_duplicate_and_non_contiguous_indices_fail(self):
        dup = [_service("worker", 0), _service("api", 0, tree=SR.NO_GGML_TREE, ld="",
                                               pid=1004)]
        self.assertEqual(
            SR.gate_stack_starts(dup, affected_services=("worker", "api")).status,
            S.FAIL)
        hole = [_service("worker", 0),
                _service("api", 5, start="2026-08-03T10:02:00+00:00",
                         ready="2026-08-03T10:03:00+00:00", tree=SR.NO_GGML_TREE,
                         ld="", pid=1005)]
        self.assertEqual(
            SR.gate_stack_starts(hole, affected_services=("worker", "api")).status,
            S.FAIL)

    def test_foreign_tree_on_ld_library_path_fails(self):
        obs = [_service("worker", 0,
                        ld="/mnt/raid0/llm/llama.cpp/build/bin:"
                           "/mnt/raid0/llm/whisper.cpp/build/src")]
        outcome = SR.gate_stack_starts(obs, affected_services=("worker",))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("another tree" in r for r in outcome.reasons))

    def test_no_ld_entry_in_its_own_tree_fails(self):
        obs = [_service("worker", 0, ld="/usr/lib")]
        outcome = SR.gate_stack_starts(obs, affected_services=("worker",))
        self.assertEqual(outcome.status, S.FAIL)

    def test_a_sibling_worktree_is_not_the_services_own_tree(self):
        # Regression, red-team 2026-08-03. Tree membership was a bare
        # `startswith`, and every sibling worktree on this host is a string
        # prefix of a declared root: `/mnt/raid0/llm/llama.cpp-experimental` (the
        # tree CLAUDE.md says ALL kernel work happens in) starts with
        # `/mnt/raid0/llm/llama.cpp`. A service linked against it read as
        # correctly linked against production and was not reported foreign —
        # gate 2 certifying the exact wrong-ggml failure it exists to catch.
        for sibling in ("/mnt/raid0/llm/llama.cpp-experimental/build/bin",
                        "/mnt/raid0/llm/llama.cpp-v5/build/bin",
                        "/mnt/raid0/llm/llama.cpp-mi210-hip/build-hip/bin"):
            outcome = SR.gate_stack_starts([_service(ld=sibling)],
                                           affected_services=("worker",))
            self.assertEqual(outcome.status, S.FAIL, sibling)
            self.assertTrue(any(sibling in r for r in outcome.reasons), sibling)

    def test_the_real_tree_root_is_still_accepted(self):
        # The guard above must not forbid the compliant path: the tree root
        # itself, and any directory under it, still satisfy gate 2.
        for own in ("/mnt/raid0/llm/llama.cpp", "/mnt/raid0/llm/llama.cpp/build/bin"):
            self.assertEqual(
                SR.gate_stack_starts([_service(ld=own)],
                                     affected_services=("worker",)).status,
                S.PASS, own)

    def test_a_treeless_service_may_not_reach_a_sibling_worktree_either(self):
        outcome = SR.gate_stack_starts(
            [_service(tree=SR.NO_GGML_TREE, linkage=None,
                      ld="/mnt/raid0/llm/llama.cpp-experimental/build/bin")],
            affected_services=("worker",))
        self.assertEqual(outcome.status, S.FAIL)

    def test_missing_linkage_receipt_is_could_not_check_not_pass(self):
        obs = [_service("worker", 0, linkage=None)]
        outcome = SR.gate_stack_starts(obs, affected_services=("worker",))
        self.assertEqual(outcome.status, S.COULD_NOT_CHECK)

    def test_linkage_from_an_unnamed_verifier_fails(self):
        receipt = SR.LinkageReceipt(verifier="/tmp/my_own_check.sh", status=S.PASS,
                                    resolved_tree="llama.cpp", receipt_ref="r")
        outcome = SR.gate_stack_starts([_service("worker", 0, linkage=receipt)],
                                       affected_services=("worker",))
        self.assertEqual(outcome.status, S.FAIL)

    def test_linkage_resolving_into_another_tree_fails(self):
        receipt = _linkage("whisper.cpp")
        outcome = SR.gate_stack_starts([_service("worker", 0, linkage=receipt)],
                                       affected_services=("worker",))
        self.assertEqual(outcome.status, S.FAIL)

    def test_failed_linkage_fails_the_gate(self):
        outcome = SR.gate_stack_starts(
            [_service("worker", 0, linkage=_linkage("llama.cpp", S.FAIL))],
            affected_services=("worker",))
        self.assertEqual(outcome.status, S.FAIL)

    def test_a_treeless_service_declares_it_and_must_not_carry_a_receipt(self):
        obs = [_service("api", 0, tree=SR.NO_GGML_TREE, ld="/usr/lib",
                        linkage=_linkage("llama.cpp"))]
        outcome = SR.gate_stack_starts(obs, affected_services=("api",))
        self.assertEqual(outcome.status, S.FAIL)

    def test_a_treeless_service_reaching_into_a_tree_fails(self):
        obs = [_service("api", 0, tree=SR.NO_GGML_TREE,
                        ld="/mnt/raid0/llm/llama.cpp/build/bin", linkage=None)]
        outcome = SR.gate_stack_starts(obs, affected_services=("api",))
        self.assertEqual(outcome.status, S.FAIL)

    def test_naive_timestamps_make_sequencing_undecidable(self):
        obs = [_service("worker", 0, start="2026-08-03T10:00:00",
                        ready="2026-08-03T10:01:00")]
        outcome = SR.gate_stack_starts(obs, affected_services=("worker",))
        self.assertEqual(outcome.status, S.COULD_NOT_CHECK)


class TestGate3LiveEqualsConfig(unittest.TestCase):
    """Gate 3: live state, never the config that was supposed to produce it."""

    def test_happy_path(self):
        live = _live()
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertEqual(outcome.status, S.PASS, outcome.reasons)

    def test_config_file_and_topology_hash_are_refused_as_live_sources(self):
        for source in ("config_file", "topology_hash", "registry", "assumed"):
            with self.assertRaises(SR.GateEvidenceMisuse):
                _live(observation_source=source)

    def test_wrong_binary_fails(self):
        live = _live(binary_sha256=_sha("some-other-binary"))
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertEqual(outcome.status, S.FAIL)

    def test_missing_and_differing_flags_fail(self):
        missing_live = _live(argv=(_PROD_BINARY, "--parallel", "8"))
        missing = SR.gate_live_equals_config([_intended()], [missing_live],
                                             started=_started(missing_live))
        self.assertEqual(missing.status, S.FAIL)

        differing_live = _live(
            argv=(_PROD_BINARY, "--parallel", "4", "--cont-batching"))
        differing = SR.gate_live_equals_config([_intended()], [differing_live],
                                               started=_started(differing_live))
        self.assertEqual(differing.status, S.FAIL)

    def test_extra_live_flag_fails_only_when_config_claims_exhaustiveness(self):
        argv = (_PROD_BINARY, "--parallel", "8", "--cont-batching", "--mlock")
        live = _live(argv=argv)
        strict = SR.gate_live_equals_config([_intended()], [live],
                                            started=_started(live))
        self.assertEqual(strict.status, S.FAIL)

        loose = SR.gate_live_equals_config(
            [_intended(flags_are_exhaustive=False)], [live], started=_started(live))
        self.assertEqual(loose.status, S.PASS)
        self.assertTrue(any("live-only flags" in n for n in loose.notes))

    def test_affinity_is_compared_against_the_live_process(self):
        live = _live(cpu_affinity=(88, 89, 90, 91))
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("affinity" in r for r in outcome.reasons))

    def test_a_process_older_than_its_config_is_stale(self):
        live = _live(started_at="2026-08-01T00:00:00+00:00")
        outcome = SR.gate_live_equals_config(
            [_intended()], [live],
            started=_started(live, start="2026-08-01T00:00:00+00:00",
                             ready="2026-08-01T00:01:00+00:00"))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("stale process" in r for r in outcome.reasons))

    def test_missing_and_extra_live_processes_fail(self):
        self.assertEqual(
            SR.gate_live_equals_config([_intended()], [],
                                       started=_started(_live())).status,
            S.FAIL)
        live, ghost = _live(), _live("ghost", pid=99)
        self.assertEqual(
            SR.gate_live_equals_config([_intended()], [live, ghost],
                                       started=_started(live, ghost)).status,
            S.FAIL)

    def test_two_live_processes_for_one_service_fail(self):
        first, second = _live(pid=1), _live(pid=2)
        outcome = SR.gate_live_equals_config([_intended()], [first, second],
                                             started=_started(second))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("leftover" in r for r in outcome.reasons))

    def test_an_empty_intended_set_cannot_pass_vacuously(self):
        with self.assertRaises(ValueError):
            SR.gate_live_equals_config([], [], started=_started(_live()))

    def test_unparseable_timestamps_make_staleness_undecidable(self):
        live = _live()
        outcome = SR.gate_live_equals_config(
            [_intended(config_recorded_at="yesterday")], [live],
            started=_started(live))
        self.assertEqual(outcome.status, S.COULD_NOT_CHECK)


class TestGate3IsTiedToGate2(unittest.TestCase):
    """AK8 carried-forward: gate 3 could pass on a process gate 2 never measured.

    Gate 2 proved that the pid it saw came up sequentially with the right ggml
    linkage; gate 3 proved that the pid IT saw matched the intended config. Nothing
    required the two pids to be the same number, let alone the same process — so
    the two green gates could describe two different processes and the §11.6 stack
    read as verified end to end.
    """

    def test_gate_2_observations_are_required_not_optional(self):
        # A keyword nobody has to pass is a tie nobody has to make.
        with self.assertRaises(TypeError):
            SR.gate_live_equals_config([_intended()], [_live()])

    def test_an_empty_gate_2_set_cannot_pass_vacuously(self):
        with self.assertRaises(ValueError):
            SR.gate_live_equals_config([_intended()], [_live()], started=())

    def test_a_non_observation_in_the_gate_2_set_is_refused(self):
        with self.assertRaises(TypeError):
            SR.gate_live_equals_config([_intended()], [_live()],
                                       started=({"service_id": "worker",
                                                 "pid": 4242},))

    def test_a_different_pid_between_the_gates_fails(self):
        live = _live(pid=4242)
        outcome = SR.gate_live_equals_config(
            [_intended()], [live], started=_started(_live(pid=7777)))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("gate 2 observed pid 7777" in r
                            for r in outcome.reasons), outcome.reasons)

    def test_a_service_with_no_gate_2_observation_fails(self):
        live = _live()
        outcome = SR.gate_live_equals_config(
            [_intended()], [live], started=_started(_live("other", pid=51)))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("no gate-2 start observation" in r
                            for r in outcome.reasons), outcome.reasons)

    def test_a_reused_pid_is_not_the_same_process(self):
        """Same number, incompatible start instant — the strongest form of the bug.

        Gate 2 watched pid 4242 come up between 09:59 and 10:05. The live process
        wearing pid 4242 started at 11:30. Matching pids alone would call that the
        same process; a pid is a reusable number.
        """
        live = _live(started_at="2026-08-03T11:30:00+00:00")
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(_live()))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("reusable number" in r for r in outcome.reasons),
                        outcome.reasons)

    def test_an_unparseable_gate_2_window_is_could_not_check_not_pass(self):
        live = _live()
        outcome = SR.gate_live_equals_config(
            [_intended()], [live],
            started=_started(live, start="whenever", ready="whenever"))
        self.assertEqual(outcome.status, S.COULD_NOT_CHECK)
        self.assertTrue(any("reused pid" in r for r in outcome.reasons),
                        outcome.reasons)

    def test_the_tie_is_recorded_in_the_evidence_reference(self):
        live = _live()
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertIn("gate2:worker#4242", outcome.evidence_ref)

    # -- compliant-path control -------------------------------------------

    def test_the_tie_admits_the_process_gate_2_actually_observed(self):
        """The guard must not forbid its own idiom: one stack, both gates.

        Same pid, live start inside the window gate 2 watched — which is what the
        normal case looks like — still PASSes, and gate 3's evidence kind is still
        its own.
        """
        live = _live()
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertEqual(outcome.status, S.PASS, outcome.reasons)
        self.assertEqual(outcome.evidence_kind, "live_process_observation")
        self.assertIn(outcome.evidence_kind, SR.GATE_EVIDENCE_KINDS[SR.GATE_3])

    def test_the_tie_admits_a_start_exactly_on_either_window_edge(self):
        for instant in ("2026-08-03T09:59:00+00:00", "2026-08-03T10:05:00+00:00"):
            live = _live(started_at=instant)
            outcome = SR.gate_live_equals_config([_intended()], [live],
                                                 started=_started(_live()))
            self.assertEqual(outcome.status, S.PASS, (instant, outcome.reasons))


class TestGate3ChecksArgv0(unittest.TestCase):
    """AK8 carried-forward: gate 3 never compared argv[0] to the binary path.

    `binary_path` comes from `proc_exe` and `argv` from `proc_cmdline` — two
    different reads. A fact that pairs one process's `exe` with another's
    `cmdline` described no process at all, and gate 3 graded it.
    """

    def test_argv0_naming_another_program_fails(self):
        live = _live(argv=("/mnt/raid0/llm/candidate/sd-server", "--parallel", "8",
                           "--cont-batching"))
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("argv[0]" in r and "different programs" in r
                            for r in outcome.reasons), outcome.reasons)

    def test_argv0_at_another_absolute_path_fails(self):
        live = _live(argv=("/mnt/raid0/llm/candidate/llama-server", "--parallel", "8",
                           "--cont-batching"))
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertEqual(outcome.status, S.FAIL)
        self.assertTrue(any("same program name, different file" in r
                            for r in outcome.reasons), outcome.reasons)

    # -- compliant-path control -------------------------------------------

    def test_argv0_equal_to_the_observed_binary_passes(self):
        live = _live()
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertEqual(outcome.status, S.PASS, outcome.reasons)
        self.assertFalse([n for n in outcome.notes if "argv[0]" in n])

    def test_a_bare_argv0_is_compared_by_program_name_and_said_so(self):
        """The compliant idiom a launcher produces: exec with a bare argv[0].

        The program name is still compared; only the directory is unavailable, and
        the outcome records which half could not be compared instead of pretending
        it was.
        """
        live = _live(argv=("llama-server", "--parallel", "8", "--cont-batching"))
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=_started(live))
        self.assertEqual(outcome.status, S.PASS, outcome.reasons)
        self.assertTrue(any("by program name only" in n for n in outcome.notes),
                        outcome.notes)


class TestMetricAndWorkloadDiscipline(unittest.TestCase):
    """task_rate, variable arrival, latency and SLO first-class."""

    def test_task_rate_cell_cannot_carry_a_token_metric(self):
        with self.assertRaises(SR.MetricSubstitutionRefused):
            SR.TaskRateCell(value=1.0, raw_samples_ref="raw", paired_blocks=1,
                            metric="decode_tokens_per_s")

    def test_metric_discipline_routes_through_the_shared_rule(self):
        self.assertEqual(
            SR.check_metric_discipline({"metric": "task_rate"}).outcome, S.PASS)
        self.assertEqual(
            SR.check_metric_discipline({"metric": "decode_tokens_s"}).outcome, S.FAIL)

    def test_fixed_shape_workload_is_refused_not_relabelled(self):
        with self.assertRaises(SR.FixedShapeWorkloadRefused):
            _workload(benchmark_class=SR.FIXED_SHAPE_BENCHMARK_CLASS)

    def test_latency_and_slo_cells_are_required_outputs(self):
        with self.assertRaises(ValueError):
            _evidence(latency=())
        with self.assertRaises(ValueError):
            _evidence(slo=())

    def test_claim_grammar_is_task_rate_and_refuses_a_kernel_protocol(self):
        evidence = _evidence()
        grammar = evidence.claim_grammar(protocol_id=SERVING_PROTOCOL, reps=12,
                                         attestation_ref="res+host+srclabel")
        self.assertEqual(grammar["metric"], "task_rate")
        self.assertEqual(grammar["category"], "CANDIDATE")
        self.assertEqual(S.check_metric_commensurability("serving_runtime",
                                                         grammar).outcome, S.PASS)
        for protocol in ("P-BENCH-1/v3", "P-GPU-1/v1", "P-BENCH-PREFILL-1"):
            with self.assertRaises(SR.MetricSubstitutionRefused):
                evidence.claim_grammar(protocol_id=protocol, reps=12,
                                       attestation_ref="x")

    def test_zero_reps_is_not_a_measurement(self):
        with self.assertRaises(ValueError):
            _evidence().claim_grammar(protocol_id=SERVING_PROTOCOL, reps=0,
                                      attestation_ref="x")

    def test_comparison_anchor_must_be_the_pinned_production_configuration(self):
        pinned = _pinned()
        self.assertEqual(SR.check_comparison_anchor(_evidence(), pinned).outcome,
                         S.PASS)

        other = SR.check_comparison_anchor(
            _evidence(comparison_config_id="scratch-config"), pinned)
        self.assertEqual(other.outcome, S.FAIL)

        moved = SR.check_comparison_anchor(
            _evidence(comparison_config_sha256=_sha("drifted")), pinned)
        self.assertEqual(moved.outcome, S.FAIL)
        self.assertTrue(any(SR.PINNED_CONFIG_MOVED in r for r in moved.reasons))

        engine = SR.check_comparison_anchor(_evidence(engine="vllm"), pinned)
        self.assertEqual(engine.outcome, S.FAIL)

        unserved = SR.check_comparison_anchor(
            _evidence(workload=_workload(trace=_trace(roles=("architect", "ghost")))),
            pinned)
        self.assertEqual(unserved.outcome, S.FAIL)


class TestObjectiveAdmission(unittest.TestCase):
    """AK-D36 excludes a target; AK-D37 keeps every regime."""

    def _objective(self, **overrides):
        objective = {
            "metric": "task_rate",
            "protocol_id": SERVING_PROTOCOL,
            "comparison_arms": [{"arm_id": "pinned", "engine": "llama-server"}],
            "recipe_class": "production_optimal",
            "concurrency_levels": [1, 16, 64],
        }
        objective.update(overrides)
        return objective

    def test_a_serving_objective_is_admitted(self):
        self.assertEqual(
            SR.admit_objective(self._objective(), pinned=_pinned()).outcome, S.PASS)

    def test_a_token_rate_objective_is_refused(self):
        check = SR.admit_objective(self._objective(metric="decode_tokens_s"),
                                   pinned=_pinned())
        self.assertEqual(check.outcome, S.FAIL)

    def test_an_objective_metric_that_is_not_task_rate_is_refused(self):
        # Regression, red-team 2026-08-03. Admission delegated the metric
        # question to `schemas.check_metric_commensurability`, which is a
        # SUBSTITUTION detector: it recognises the token-rate spellings it knows
        # and passes everything else. `tok/s` and `t/s` — the spellings this
        # project's own bench records and llama-bench output use — cleared it, so
        # a token-rate objective was admissible as long as it was spelled the way
        # we actually spell it. So was every unrelated metric.
        for metric in ("tok/s", "t/s", "ms_per_token", "prompt_tok_s",
                       "whatever_i_like"):
            check = SR.admit_objective(self._objective(metric=metric),
                                       pinned=_pinned())
            self.assertEqual(check.outcome, S.FAIL, metric)
            self.assertTrue(any("task_rate" in r for r in check.reasons), metric)
        self.assertEqual(
            SR.admit_objective(self._objective(metric=SR.METRIC),
                               pinned=_pinned()).outcome, S.PASS)

    def test_a_cross_engine_arm_is_refused_as_a_target(self):
        check = SR.admit_objective(
            self._objective(comparison_arms=[{"arm_id": "vllm", "engine": "vllm"}]),
            pinned=_pinned())
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("AK-D36" in r for r in check.reasons))

    def test_a_kernel_protocol_objective_is_refused(self):
        check = SR.admit_objective(self._objective(protocol_id="P-BENCH-1/v3"),
                                   pinned=_pinned())
        self.assertEqual(check.outcome, S.FAIL)

    def test_an_off_recipe_objective_is_refused(self):
        check = SR.admit_objective(self._objective(recipe_class="baseline"),
                                   pinned=_pinned())
        self.assertEqual(check.outcome, S.FAIL)

    def test_every_batch_regime_stays_admissible(self):
        # AK-D37: the constraint is on the metric, never on the batch regime.
        for level in (1, 2, 8, 16, 64, 128, 512):
            self.assertEqual(SR.check_regime_admissible(level).outcome, S.PASS, level)
            self.assertEqual(
                SR.admit_objective(self._objective(concurrency_levels=[level]),
                                   pinned=_pinned()).outcome, S.PASS, level)
        self.assertEqual(_workload(concurrency=1).concurrency, 1)
        self.assertEqual(_workload(concurrency=64).concurrency, 64)

    def test_nonsense_concurrency_is_still_rejected(self):
        self.assertEqual(SR.check_regime_admissible(0).outcome, S.FAIL)
        self.assertEqual(SR.check_regime_admissible("many").outcome, S.FAIL)

    def test_missing_arms_or_metric_is_could_not_check(self):
        objective = self._objective()
        del objective["comparison_arms"]
        self.assertEqual(SR.admit_objective(objective, pinned=_pinned()).outcome,
                         S.COULD_NOT_CHECK)
        self.assertEqual(SR.admit_objective({}, pinned=_pinned()).outcome,
                         S.COULD_NOT_CHECK)

    def test_cross_engine_view_is_reportable_but_never_gating(self):
        view = SR.CrossEngineAnalysisView(
            label="whole-stack ratio, analysis only", incumbent_engine="llama-server",
            comparator_engine="vllm", concurrency=64, observed_ratio=24.0)
        self.assertFalse(view.gates)
        self.assertIn("AK-D36", view.note)
        with self.assertRaises(SR.CrossEngineRatioObjectiveRefused):
            SR.CrossEngineAnalysisView(
                label="x", incumbent_engine="llama-server", comparator_engine="vllm",
                concurrency=64, observed_ratio=24.0, gates=True)


class TestChangeClassAdmission(unittest.TestCase):
    """Admission is derived from the cheap-suite mapping, not hand-listed."""

    def test_only_variable_arrival_replay_classes_are_admitted(self):
        expected = {
            name for name, suite in S.CHANGE_CLASS_CHEAP_SUITE.items()
            if suite == SR.WORKLOAD_CLASS
        }
        admitted = set()
        for name in S.CHANGE_CLASSES:
            try:
                SR.admit_change_class(name)
            except ValueError:
                continue
            admitted.add(name)
        self.assertEqual(admitted, expected)
        self.assertIn("scheduler_policy", expected)

    def test_kernel_change_classes_are_refused(self):
        for name in ("arithmetic", "layout", "fusion", "core_header", "parameter"):
            with self.assertRaises(ValueError):
                SR.admit_change_class(name)

    def test_oracle_port_defers_to_its_underlying_class(self):
        self.assertEqual(
            SR.admit_change_class("oracle_port", underlying="scheduler_policy"),
            "oracle_port")
        with self.assertRaises(ValueError):
            SR.admit_change_class("oracle_port")
        with self.assertRaises(ValueError):
            SR.admit_change_class("oracle_port", underlying="arithmetic")

    def test_unknown_class_is_refused(self):
        with self.assertRaises(ValueError):
            SR.admit_change_class("vibes")


class TestT2Composition(unittest.TestCase):
    """§9.7: composed champions are measured, never added."""

    def _composition(self, **overrides):
        kwargs = dict(
            composition_id="comp-1",
            member_candidate_ids=("akc-a", "akc-b"),
            evidence=_evidence(),
            measured_as_whole=True,
            window_ref="window://t2/1",
        )
        kwargs.update(overrides)
        return SR.ComposedServingEstimate(**kwargs)

    def test_a_measured_composition_is_accepted(self):
        self.assertEqual(len(self._composition().member_candidate_ids), 2)

    def test_additive_construction_is_refused_where_it_would_be_looked_for(self):
        with self.assertRaises(SR.AdditiveCompositionRefused):
            SR.ComposedServingEstimate.from_local_effects(0.03, 0.02)

    def test_an_unmeasured_composition_is_refused(self):
        with self.assertRaises(SR.AdditiveCompositionRefused):
            self._composition(measured_as_whole=False)

    def test_a_composition_needs_at_least_two_distinct_members(self):
        with self.assertRaises(ValueError):
            self._composition(member_candidate_ids=("akc-a",))
        with self.assertRaises(ValueError):
            self._composition(member_candidate_ids=("akc-a", "akc-a"))

    def test_a_composition_needs_sentinel_breadth(self):
        narrow = _evidence(
            workload=_workload(trace=_trace(roles=("architect",))),
            slo=(SR.SloCell(slo_id="only", target_description="p95 <= 3000 ms",
                            attainment=0.99, window_s=600.0,
                            raw_samples_ref="raw://slo"),),
        )
        with self.assertRaises(ValueError):
            self._composition(evidence=narrow)


class TestComplexityCeiling(unittest.TestCase):
    """§10.6: mark for review; never truncate, never silently accept."""

    def test_under_the_ceiling_needs_no_marker(self):
        radius = SR.classify_blast_radius(diff_lines=100, files_touched=3,
                                          touches_shared_core=False)
        self.assertFalse(radius.requires_human_code_review)

    def test_above_the_ceiling_is_marked_not_rejected(self):
        radius = SR.classify_blast_radius(diff_lines=5000, files_touched=3,
                                          touches_shared_core=False)
        self.assertTrue(radius.requires_human_code_review)
        self.assertEqual(radius.diff_lines, 5000)  # not capped, not truncated

        by_files = SR.classify_blast_radius(diff_lines=10, files_touched=99,
                                            touches_shared_core=False)
        self.assertTrue(by_files.requires_human_code_review)

    def test_the_ceiling_forbids_shared_core_at_any_size(self):
        self.assertFalse(SR.SERVING_COMPLEXITY_CEILING.shared_core_permitted)


class TestStackChangePackage(unittest.TestCase):
    """What the adapter hands a human, and what it refuses to hand them."""

    def test_a_green_package_is_pass(self):
        package = _package()
        self.assertEqual(package.verdict, "PASS")
        self.assertEqual(package.to_record()["backend"], "serving_runtime")
        self.assertEqual(package.to_record()["release_path"], "stack_change")

    def test_package_id_namespace_is_distinct_from_the_kernel_package(self):
        with self.assertRaises(ValueError):
            _package(package_id="akr-serving-1")

    def test_a_waiver_makes_it_pass_with_waiver_and_must_suppress_a_claim(self):
        package = _package(active_waivers=(_waiver(),),
                           suppressed_claims=("p99 non-regression for gemma4-26B-A4B",))
        self.assertEqual(package.verdict, "PASS_WITH_WAIVER")
        with self.assertRaises(ValueError):
            _package(active_waivers=(_waiver(),))
        with self.assertRaises(ValueError):
            _package(suppressed_claims=("nothing was waived",))

    def test_a_waiver_from_another_campaign_is_refused(self):
        with self.assertRaises(ValueError):
            _package(active_waivers=(_waiver(campaign_id="ak-some-other-campaign"),),
                     suppressed_claims=("x",))

    def test_an_invalid_waiver_is_refused(self):
        broken = _waiver()
        del broken["consequences"]
        with self.assertRaises(ValueError):
            _package(active_waivers=(broken,), suppressed_claims=("x",))

    def test_a_waiver_never_rescues_a_failed_gate(self):
        gates = SR.evaluate_three_gates(
            pipeline_green=_gate(SR.GATE_1),
            stack_starts=_gate(SR.GATE_2, status=S.FAIL, reasons=("api never came up",)),
        )
        package = _package(gates=gates, active_waivers=(_waiver(),),
                           suppressed_claims=("x",))
        self.assertEqual(package.verdict, "FAIL")

    def test_a_verdict_cannot_be_attached_that_the_gates_do_not_support(self):
        with self.assertRaises(SR.PackageTampering):
            SR.StackChangePackage(
                package_id="aks-1", campaign_id=CAMPAIGN, verdict="PASS",
                gates=SR.evaluate_three_gates(
                    pipeline_green=_gate(SR.GATE_1, status=S.FAIL, reasons=("red",))),
                evidence=_evidence(), pinned=_pinned(), rollback=_rollback(),
                blast_radius=SR.classify_blast_radius(diff_lines=1, files_touched=1,
                                                      touches_shared_core=False),
                operator_command_sequence=_commands(),
                created_at="2026-08-03T12:00:00+00:00",
            )

    def test_every_operator_command_must_be_pre_validated(self):
        with self.assertRaises(ValueError):
            _package(operator_command_sequence=(
                {"command": "orchestrator_stack.py reload orchestrator",
                 "validated": False, "validation_receipt": "none"},))
        with self.assertRaises(ValueError):
            _package(operator_command_sequence=())

    def test_an_unverified_rollback_archive_is_not_a_rollback(self):
        with self.assertRaises(ValueError):
            _rollback(archive_verified=False)

    def test_a_naive_created_at_is_refused(self):
        with self.assertRaises(ValueError):
            _package(created_at="2026-08-03T12:00:00")

    def test_review_marker_must_ride_along_when_the_ceiling_is_exceeded(self):
        package = _package(blast_radius=SR.classify_blast_radius(
            diff_lines=9000, files_touched=40, touches_shared_core=False))
        self.assertTrue(package.requires_human_code_review)
        self.assertTrue(package.to_record()["requires_human_code_review"])

    def test_the_record_carries_no_authority_flavoured_key(self):
        record = _package(active_waivers=(_waiver(),),
                          suppressed_claims=("x",)).to_record()
        self.assertEqual(S.find_authority_flavoured_keys(record), [])

    def test_the_record_is_canonically_serialisable(self):
        record = _package().to_record()
        reloaded = json.loads(S.canonical_json(record))
        self.assertEqual(reloaded["backend"], "serving_runtime")
        self.assertEqual(reloaded["metric"], "task_rate")
        self.assertEqual(reloaded["verdict"], "PASS")
        # Content-hashable, which is what the release plane consumes.
        self.assertRegex(S.content_hash(record), r"^[0-9a-f]{64}$")

    def test_assembly_refuses_a_comparison_that_is_not_the_pinned_config(self):
        with self.assertRaises(SR.ServingAdapterError):
            _package(evidence=_evidence(comparison_config_id="scratch"))

    def test_a_directly_constructed_package_checks_its_anchor_too(self):
        # Regression, red-team 2026-08-03. The anchor refusal lived only in
        # `assemble_stack_change_package()`, but `StackChangePackage` is exported
        # and constructible, and it is the durable artifact. A refusal present in
        # one of two constructors is a refusal a caller passes by using the other
        # one, so the package now re-runs the check on its own two fields.
        def build(**overrides):
            kwargs = dict(
                package_id="aks-direct-1", campaign_id=CAMPAIGN, verdict="PASS",
                gates=_all_gates_pass(), evidence=_evidence(), pinned=_pinned(),
                rollback=_rollback(),
                blast_radius=SR.classify_blast_radius(
                    diff_lines=10, files_touched=1, touches_shared_core=False),
                operator_command_sequence=_commands(),
                created_at="2026-08-03T12:00:00+00:00")
            kwargs.update(overrides)
            return SR.StackChangePackage(**kwargs)

        build()  # the compliant path still constructs
        for bad in (
            _evidence(comparison_config_id="scratch"),
            _evidence(comparison_config_sha256=_sha("a config that has moved")),
            _evidence(engine="vllm"),
        ):
            with self.assertRaises(SR.ServingAdapterError):
                build(evidence=bad)

    def test_the_record_carries_the_instrument_and_the_raw_sample_pointers(self):
        # Regression, red-team 2026-08-03. The record carried the task_rate NUMBER
        # and neither the candidate it belongs to, the replay that produced it,
        # nor any raw-samples pointer — a durable artifact from which the
        # measurement could not be found again, let alone replayed.
        evidence = _evidence()
        record = _package(evidence=evidence).to_record()
        self.assertEqual(record["candidate_id"], evidence.candidate_id)
        self.assertEqual(record["workload"]["trace"]["trace_sha256"],
                         evidence.workload.trace.trace_sha256)
        self.assertEqual(record["workload"]["trace"]["seed"],
                         evidence.workload.trace.seed)
        self.assertEqual(record["workload"]["concurrency"],
                         evidence.workload.concurrency)
        self.assertEqual(record["workload"]["recipe_sha256"],
                         evidence.workload.recipe_sha256)
        self.assertEqual(record["task_rate_cell"]["raw_samples_ref"],
                         evidence.task_rate.raw_samples_ref)
        self.assertEqual(record["task_rate_cell"]["paired_blocks"],
                         evidence.task_rate.paired_blocks)
        for cell, written in zip(evidence.latency, record["latency_cells"]):
            self.assertEqual(written["raw_samples_ref"], cell.raw_samples_ref)
        for cell, written in zip(evidence.slo, record["slo_cells"]):
            self.assertEqual(written["raw_samples_ref"], cell.raw_samples_ref)
            self.assertEqual(written["window_s"], cell.window_s)
        # Still clean under the two scans the record must survive.
        self.assertEqual(S.find_authority_flavoured_keys(record), [])
        self.assertRegex(S.content_hash(record), r"^[0-9a-f]{64}$")


class TestReloadRequest(unittest.TestCase):
    """§11.3: a reload is a routed request, and the pinned bench is the gate."""

    def test_a_reload_request_is_a_record_bound_to_the_pinned_bench(self):
        request = SR.build_reload_request(
            request_id="req-1", owning_session="mainB", services=("worker",),
            pinned=_pinned(), gates=_all_gates_pass(), autopilot_state="down")
        self.assertEqual(request.gate_basis, SR.RELOAD_GATE_BASIS)
        self.assertEqual(request.pinned_bench_receipt, "bench://cpu-pinned/20260801")
        self.assertTrue(any("not a gate" in n for n in request.notes))

    def test_autopilot_state_cannot_be_the_basis(self):
        with self.assertRaises(SR.ReloadGateBasisRefused):
            SR.build_reload_request(
                request_id="req-2", owning_session="mainB", services=("worker",),
                pinned=_pinned(), gates=_all_gates_pass(), gate_basis="autopilot_down")

    def test_no_reload_request_without_three_green_gates(self):
        gates = SR.evaluate_three_gates(pipeline_green=_gate(SR.GATE_1))
        with self.assertRaises(SR.ServingAdapterError):
            SR.build_reload_request(
                request_id="req-3", owning_session="mainB", services=("worker",),
                pinned=_pinned(), gates=gates)


class TestEvaluationEvent(unittest.TestCase):
    """A serving cell must be journal-able under the shared v3 contract."""

    def _event(self, **overrides):
        kwargs = dict(
            event_id="ake-serving-0001",
            candidate_id=CANDIDATE,
            tier="T2",
            evidence=_evidence(),
            pinned=_pinned(),
            protocol_id=SERVING_PROTOCOL,
            reps=12,
            attestation_ref="res=claim-1;host=host-1;srclabel=src-1",
            evaluator_id="autokernel.evaluator/v1",
            evaluator_bundle_sha256=_sha("bundle"),
            artifact_source_sha256=_sha("candidate-config"),
            anchor_source_commit=_commit("orchestrator-head"),
            anchor_measurement_event_ids=["ake-anchor-0001"],
            scope_manifest_sha256=_sha("scope"),
            host_receipt="host://health/20260803",
            resource_claim_receipt="claim://stack/20260803",
            scope_denominator={"machine_subset": "full", "numa_nodes": [0, 1, 2, 3],
                               "devices": [], "cores": 96},
            correctness={"replayed_requests_ok": 4096, "http_5xx": 0},
            quality={"answer_parity_sampled": 128},
            stability={"restarts": 0, "rss_growth_mb": 12},
            mechanism={"admission_queue_depth_p95": 3},
            performance={"raw_samples": [3.2, 3.3, 3.1], "paired_blocks": 12,
                         "estimate": 0.041, "uncertainty": 0.011},
            determinism={"class": "not_measured", "same_seed_repeat_runs": 0},
            status="pass",
            created_at="2026-08-03T12:30:00+00:00",
        )
        kwargs.update(overrides)
        return SR.build_serving_evaluation_event(**kwargs)

    def test_the_event_validates_under_the_shared_schema(self):
        event = self._event()
        self.assertEqual(S.validate_evaluation_event(event), [])
        self.assertEqual(event["claim_grammar"]["metric"], "task_rate")

    def test_the_anchor_is_the_pinned_production_kernel(self):
        event = self._event()
        pinned = _pinned()
        self.assertEqual(event["anchor"]["binary_sha256"], pinned.kernel_binary_sha256)
        # Identical across arms by construction: that is what makes it a
        # scheduler comparison rather than a kernel one.
        self.assertEqual(event["artifact"]["binary_sha256"],
                         event["anchor"]["binary_sha256"])

    def test_co_residency_is_derived_from_the_replay(self):
        multi = self._event()
        self.assertTrue(multi["co_residency"].startswith("co_resident:"))
        single = self._event(
            evidence=_evidence(workload=_workload(trace=_trace(roles=("architect",)))))
        self.assertEqual(single["co_residency"], "single")

    def test_a_contradicting_co_residency_is_refused(self):
        with self.assertRaises(SR.ServingAdapterError):
            self._event(co_residency="single")

    def test_the_performance_block_must_be_the_evidence_cells_measurement(self):
        # Regression, red-team 2026-08-03. `performance` is caller-supplied and
        # the `evidence` argument never reached it: the event's numbers and the
        # cell its claim grammar vouches for could come from different
        # measurements entirely, and nothing said so. `paired_blocks` is the one
        # field that admits no second reading.
        self._event()  # the compliant path still builds
        with self.assertRaises(SR.ServingAdapterError):
            self._event(performance={"raw_samples": [3.2, 3.3, 3.1],
                                     "paired_blocks": 3, "estimate": 0.041,
                                     "uncertainty": 0.011})

    def test_a_kernel_protocol_cannot_be_cited(self):
        with self.assertRaises(SR.MetricSubstitutionRefused):
            self._event(protocol_id="P-BENCH-1/v3")

    def test_an_invalid_event_raises_instead_of_being_emitted(self):
        with self.assertRaises(SR.ServingAdapterError):
            self._event(event_id="serving-0001")  # missing the 'ake-' prefix
        with self.assertRaises(SR.ServingAdapterError):
            self._event(performance={"raw_samples": [], "paired_blocks": 0,
                                     "estimate": 0.04, "uncertainty": 0.01})

    def test_the_event_carries_no_authority_flavoured_key(self):
        self.assertEqual(S.find_authority_flavoured_keys(self._event()), [])


class TestNoProcessOrWritePaths(unittest.TestCase):
    """The no-write, no-spawn, no-signal property, checked from the AST."""

    def test_module_cannot_write_spawn_or_signal(self):
        self.assertEqual(SR.audit_no_write_or_process_paths().outcome, S.PASS)

    def test_the_audit_actually_fires(self):
        # A fixture that removes the signal under test would pass a broken audit.
        bad = "import subprocess\ndef go():\n    subprocess.Popen(['x'])\n"
        result = SR.audit_no_write_or_process_paths(bad)
        self.assertEqual(result.outcome, S.FAIL)
        self.assertEqual(len(result.reasons), 2)

    def test_the_audit_is_not_passed_by_giving_it_nothing_to_audit(self):
        # Regression, red-team 2026-08-03. An empty source parsed fine, contained
        # no forbidden call, and returned PASS — the check certifying its own
        # absence, which is exactly the "delete the thing it inspects" pass.
        for empty in ("", "\n\n", "# only a comment\n"):
            self.assertEqual(SR.audit_no_write_or_process_paths(empty).outcome,
                             S.COULD_NOT_CHECK, repr(empty))

    def test_the_audit_sees_the_pathlib_write_link_and_import_verbs(self):
        # Regression, red-team 2026-08-03. `open` was refused only as a BARE
        # name, so `Path(p).open("w")` — the idiomatic write door — passed. The
        # link verbs are spelled `symlink_to`/`hardlink_to` in pathlib while the
        # audit named `symlink`/`link`, so the call that performs
        # `move_stable_kernel_symlink` (item one on FORBIDDEN_PRODUCTION_ACTIONS)
        # was invisible. `__import__` was refused but `importlib.import_module`
        # was not, which reopened every banned module.
        for source in (
            "from pathlib import Path\ndef go():\n    Path('/x').open('w')\n",
            "from pathlib import Path\ndef go():\n    Path('/a').symlink_to('/b')\n",
            "from pathlib import Path\ndef go():\n    Path('/a').hardlink_to('/b')\n",
            "import importlib\ndef go():\n    importlib.import_module('os')\n",
        ):
            result = SR.audit_no_write_or_process_paths(source)
            self.assertEqual(result.outcome, S.FAIL, source)

    def test_the_audit_does_not_fire_on_ordinary_reading_code(self):
        # The guard must not forbid the compliant idiom: this module reads its
        # own source with `Path.read_text()` and formats strings, and neither is
        # a write. So the clean snippet must not FAIL — it is COULD_NOT_CHECK
        # rather than PASS only because it is not THIS module's source, which is
        # the anchoring the two sibling adapters already had and this one did not.
        clean = ("from pathlib import Path\n"
                 "def go(p):\n"
                 "    text = Path(p).read_text(encoding='utf-8')\n"
                 "    return text.replace('a', 'b').strip().split(':')\n")
        self.assertNotEqual(SR.audit_no_write_or_process_paths(clean).outcome, S.FAIL)
        # …and the real module, which uses exactly those idioms, still PASSes.
        own = Path(SR.__file__).read_text(encoding="utf-8")
        check = SR.audit_no_write_or_process_paths(own)
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_a_clean_foreign_module_is_not_this_modules_clean_bill_of_health(self):
        """The `source=` seam was the last unbound audit in the adapter plane.

        `whisper_stt` and `qwentts_tts` bind theirs to their own module identity;
        this one returned PASS for any clean text, so auditing a sibling adapter —
        or the empty string with one `def` in it — read as evidence about this
        module.
        """
        for text in ("x = 1\n", "def f():\n    return 1\n",
                     "from pathlib import Path\ndef g(p):\n    return Path(p).read_text()\n"):
            check = SR.audit_no_write_or_process_paths(text)
            self.assertEqual(check.outcome, S.COULD_NOT_CHECK, text)
            self.assertIn("identity", " ".join(check.reasons))

    def test_a_forbidden_construct_still_fails_even_unbound(self):
        # A FAIL is a finding about the text whoever the text belongs to, so the
        # binding must not convert a positive detection into COULD_NOT_CHECK.
        self.assertEqual(
            SR.audit_no_write_or_process_paths("p.write_text('x')\n").outcome, S.FAIL)

    def test_unparseable_source_is_could_not_check(self):
        self.assertEqual(
            SR.audit_no_write_or_process_paths("def (:").outcome, S.COULD_NOT_CHECK)

    def test_the_module_does_not_import_the_orchestrator_repo(self):
        source = Path(SR.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                self.assertNotIn("orchestrator", stripped)
                self.assertNotIn("src.registry", stripped)

    def test_the_module_exposes_no_start_stop_or_kill_verb(self):
        for name in dir(SR):
            self.assertNotIn(name.split("_")[0], {"start", "stop", "kill", "restart",
                                                  "signal", "spawn", "launch"})


class TestStableKernelPathDepthCannotBeWalkedOut(unittest.TestCase):
    """Red-team of the executes-vs-mutates classifier: the DEPTH premise.

    The classifier decides execution from two facts, and the first is depth —
    `…/kernels/production/<lane>/<x>/…` is content reached THROUGH the link, so
    it may stand in command-word position. Two path spellings defeated the
    measurement, and both of them are ordinary shell, not exotica.
    """

    def test_a_parent_traversal_out_of_the_stable_path_is_refused(self):
        """`..` counted as depth, so any host binary could wear the stable path.

        `…/production/cpu/../../../../usr/bin/install` counts seven components —
        "well below the link" — while naming `/usr/bin/install`. Admitted as the
        command word it also flipped the segment to "a production binary is
        running", which admitted the segment's below-link OPERANDS, and the
        operand here is the production serving binary being overwritten.
        """
        escape = _PROD_LANE + "/../../../../usr/bin/install"
        command = escape + " -m 0755 /tmp/evil " + _PROD_LANE + "/bin/llama-server"
        check = SR.classify_stable_kernel_path_use(command)
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        self.assertTrue(any("'..'" in r for r in check.reasons), check.reasons)
        self.assertEqual(
            SR.scan_for_kernel_freeze_actions({"cmd": command}).outcome, S.FAIL)

    def test_every_traversal_shape_is_refused(self):
        cases = {
            "escape to rm": _PROD_LANE + "/../../../../bin/rm -rf " + _PROD_LANE,
            "escape mid-path": (_PROD_LANE + "/bin/../../../../bin/cp /tmp/x "
                                + _PROD_BINARY),
            "traversal as an operand": "rm -rf " + _PROD_LANE + "/bin/..",
            "traversal in a loader path": ("LD_LIBRARY_PATH=" + _PROD_LANE
                                          + "/../../../etc " + _PROD_BINARY),
        }
        for label, command in cases.items():
            with self.subTest(label):
                self.assertEqual(SR.classify_stable_kernel_path_use(command).outcome,
                                 S.FAIL, command)
                self.assertEqual(
                    SR.scan_for_kernel_freeze_actions({"cmd": command}).outcome,
                    S.FAIL, command)

    def test_a_repeated_slash_does_not_hide_the_stable_path(self):
        """`a//b` is `a/b` to every kernel on this host.

        The segment search was a plain substring find, so one extra slash made a
        straight `rm -rf` of the production tree produce no finding AT ALL — not a
        weaker finding, not COULD_NOT_CHECK, nothing.
        """
        doubled = "/mnt/raid0/llm/kernels//production/cpu"
        for command in ("rm -rf " + doubled + "/bin/llama-server",
                        "ln -sfn /mnt/raid0/llm/llama.cpp-v9 " + doubled):
            with self.subTest(command):
                self.assertEqual(SR.classify_stable_kernel_path_use(command).outcome,
                                 S.FAIL, command)
                self.assertEqual(
                    SR.scan_for_kernel_freeze_actions({"cmd": command}).outcome,
                    S.FAIL, command)

    def test_a_traversing_command_cannot_be_packaged(self):
        commands = (
            {"command": (_PROD_LANE + "/../../../../usr/bin/install -m 0755 /tmp/x "
                         + _PROD_BINARY),
             "validated": True, "validation_receipt": "dryrun://install"},
        )
        with self.assertRaises(SR.KernelFreezePathRefused):
            _package(operator_command_sequence=commands)

    # -- compliant-path controls ------------------------------------------

    def test_the_launcher_idiom_is_still_admitted_unchanged(self):
        """The depth guard must not forbid the idiom depth exists to permit."""
        cases = {
            "bare": _PROD_BINARY + " --port 8080",
            "taskset": "taskset -c 184-191 " + _PROD_BINARY + " -t 96 -fa 1",
            "own LD_LIBRARY_PATH": ("LD_LIBRARY_PATH=" + _PROD_LANE + "/lib "
                                    + _PROD_BINARY + " --port 8080"),
            "its own tree as an operand": (
                _PROD_BINARY + " --lora " + _PROD_LANE + "/share/adapter.gguf"),
        }
        for label, command in cases.items():
            with self.subTest(label):
                self.assertEqual(SR.classify_stable_kernel_path_use(command).outcome,
                                 S.PASS, command)

    def test_a_traversal_that_is_nowhere_near_production_is_not_forbidden(self):
        """`..` is banned BELOW the stable path, not banned from the command.

        A guard that refused every `..` anywhere would refuse `-m ../models/x.gguf`
        — a relative model path is ordinary — and would be the same defect this
        class exists to close, pointing the other way.
        """
        command = ("taskset -c 0-95 " + _PROD_BINARY
                   + " -m ../models/qwen36-q8.gguf --log ../logs/serve.log")
        self.assertEqual(SR.classify_stable_kernel_path_use(command).outcome,
                         S.PASS)
        self.assertEqual(
            SR.scan_for_kernel_freeze_actions({"cmd": command}).outcome, S.PASS)

    def test_a_doubled_slash_elsewhere_is_not_read_as_the_stable_path(self):
        command = _PROD_BINARY + " --model-url https://example.invalid/m.gguf"
        self.assertEqual(SR.classify_stable_kernel_path_use(command).outcome,
                         S.PASS, command)


class TestGate3RefusesAnAmbiguousGate2Set(unittest.TestCase):
    """Red-team of the gate-2/gate-3 tie: the gate-2 side was collapsed.

    `started_by_id[obs.service_id] = obs` in a plain loop is last-wins. Gate 2
    FAILs a service that "appears twice in one observation set" and the live side
    of gate 3 FAILs two live processes for one service; the gate-2 side of the
    tie silently kept whichever observation came last.
    """

    def test_two_gate2_observations_for_one_service_are_a_finding(self):
        live = _live()
        stale = _service(service_id="worker", pid=1000)
        current = _service(service_id="worker", pid=4242,
                           start="2026-08-03T09:59:00+00:00",
                           ready="2026-08-03T10:05:00+00:00")
        outcome = SR.gate_live_equals_config([_intended()], [live],
                                             started=[stale, current])
        self.assertEqual(outcome.status, S.FAIL, outcome.notes)
        self.assertTrue(any("two gate-2 start observations" in r
                            for r in outcome.reasons), outcome.reasons)

    def test_the_verdict_does_not_depend_on_the_order_the_evidence_was_listed(self):
        """The bite: the SAME evidence in two orders gave PASS and FAIL."""
        live = _live()
        stale = _service(service_id="worker", pid=1000)
        current = _service(service_id="worker", pid=4242,
                           start="2026-08-03T09:59:00+00:00",
                           ready="2026-08-03T10:05:00+00:00")
        forward = SR.gate_live_equals_config([_intended()], [live],
                                             started=[stale, current])
        reverse = SR.gate_live_equals_config([_intended()], [live],
                                             started=[current, stale])
        self.assertEqual(forward.status, S.FAIL)
        self.assertEqual(reverse.status, S.FAIL)

    def test_every_observed_pid_is_named_in_the_evidence_reference(self):
        live = _live()
        outcome = SR.gate_live_equals_config(
            [_intended()], [live],
            started=[_service(service_id="worker", pid=1000),
                     _service(service_id="worker", pid=4242)])
        self.assertIn("worker#1000+4242", outcome.evidence_ref)

    # -- compliant-path control -------------------------------------------

    def test_one_observation_per_service_is_still_admitted(self):
        """Several services, one gate-2 observation each: the normal stack."""
        facts = [_live(service_id="worker"), _live(service_id="router", pid=4343)]
        intended = [_intended(service_id="worker"), _intended(service_id="router")]
        outcome = SR.gate_live_equals_config(intended, facts,
                                             started=_started(*facts))
        self.assertEqual(outcome.status, S.PASS, outcome.reasons)


class TestReleaseVerdictRequiresTheGate2Binding(unittest.TestCase):
    """Red-team of the tie's COMPOSITION: it lived only inside the function.

    A caller assembling a verdict holds `GateOutcome`s — from a journal, from a
    prior run, from wherever. `evaluate_three_gates()` derived `released == True`
    from a gate-3 outcome that declared no tie at all, and from one citing
    processes the gate-2 outcome beside it had never observed. A guarantee
    obtainable by not calling the function that provides it is not a guarantee.
    """

    def test_an_untied_gate3_cannot_produce_a_release_verdict(self):
        with self.assertRaises(SR.GateSubjectUnbound):
            SR.evaluate_three_gates(
                pipeline_green=_gate(SR.GATE_1),
                stack_starts=_gate(SR.GATE_2),
                live_equals_config=_gate(SR.GATE_3, tied_to=()),
            )

    def test_a_gate3_tied_to_processes_gate2_never_observed_is_refused(self):
        with self.assertRaises(SR.GateSubjectUnbound) as ctx:
            SR.evaluate_three_gates(
                pipeline_green=_gate(SR.GATE_1),
                stack_starts=_gate(SR.GATE_2, subjects=("worker#4242",)),
                live_equals_config=_gate(SR.GATE_3, tied_to=("router#77",)),
            )
        self.assertIn("different processes", str(ctx.exception))

    def test_a_pid_reused_between_the_gates_is_refused(self):
        # Same service, different process: the tie is on `service#pid`, so the
        # replacement is visible at the verdict, not only inside gate 3.
        with self.assertRaises(SR.GateSubjectUnbound):
            SR.evaluate_three_gates(
                pipeline_green=_gate(SR.GATE_1),
                stack_starts=_gate(SR.GATE_2, subjects=("worker#1000",)),
                live_equals_config=_gate(SR.GATE_3, tied_to=("worker#4242",)),
            )

    def test_the_binding_is_not_satisfiable_by_supplying_neither_side(self):
        """Two empty tuples compare equal; a tie that only tested equality was
        passed by DELETING both sides of it."""
        with self.assertRaises(SR.GateSubjectUnbound):
            SR.evaluate_three_gates(
                pipeline_green=_gate(SR.GATE_1),
                stack_starts=_gate(SR.GATE_2, subjects=()),
                live_equals_config=_gate(SR.GATE_3, tied_to=()),
            )

    def test_the_binding_cannot_be_skipped_by_building_the_result_directly(self):
        # `ThreeGateResult` is constructible; the verdict-tampering check already
        # lives in its `__post_init__` for exactly that reason, and so does this.
        with self.assertRaises(SR.GateSubjectUnbound):
            SR.ThreeGateResult(
                pipeline_green=_gate(SR.GATE_1),
                stack_starts=_gate(SR.GATE_2),
                live_equals_config=_gate(SR.GATE_3, tied_to=()),
                status=S.PASS, blocked_at=None,
            )

    def test_a_package_cannot_be_assembled_on_an_unbound_verdict(self):
        with self.assertRaises(SR.GateSubjectUnbound):
            _package(gates=SR.evaluate_three_gates(
                pipeline_green=_gate(SR.GATE_1),
                stack_starts=_gate(SR.GATE_2),
                live_equals_config=_gate(SR.GATE_3, tied_to=()),
            ))

    def test_gate1_cannot_declare_a_binding_to_something_that_precedes_it(self):
        with self.assertRaises(ValueError):
            SR.GateOutcome(gate=SR.GATE_1, status=S.PASS,
                           evidence_kind="stack_change_guard_result",
                           evidence_ref="guard", tied_to=("worker#4242",))

    # -- compliant-path control -------------------------------------------

    def test_the_real_gate_functions_compose_into_a_released_verdict(self):
        """The idiom the binding exists to permit: gates 2 and 3 as actually run.

        Both gate functions are called on ONE observation set, and the verdict
        assembles and is release-eligible. A binding that the module's own gate
        pair could not satisfy would be the `feedback_guard_must_not_forbid_its_
        own_idiom` defect over again.
        """
        # Sequential starts, because gate 2 requires them: worker comes up, then
        # router (`feedback_sequential_model_loading`).
        facts = [_live(service_id="worker"),
                 _live(service_id="router", pid=4343,
                       started_at="2026-08-03T10:01:30+00:00")]
        observations = (
            _service(service_id="worker", index=0, pid=4242,
                     start="2026-08-03T09:59:00+00:00",
                     ready="2026-08-03T10:00:30+00:00"),
            _service(service_id="router", index=1, pid=4343,
                     start="2026-08-03T10:01:00+00:00",
                     ready="2026-08-03T10:02:00+00:00"),
        )
        gate2 = SR.gate_stack_starts(
            observations, affected_services=("worker", "router"))
        self.assertEqual(gate2.status, S.PASS, gate2.reasons)
        gate3 = SR.gate_live_equals_config(
            [_intended(service_id="worker"), _intended(service_id="router")],
            facts, started=observations)
        self.assertEqual(gate3.status, S.PASS, gate3.reasons)
        result = SR.evaluate_three_gates(pipeline_green=_gate(SR.GATE_1),
                                         stack_starts=gate2,
                                         live_equals_config=gate3)
        self.assertTrue(result.released)
        result.require_release_eligible()

    def test_the_binding_is_still_derived_from_observations_not_intentions(self):
        """Gate 2's subjects come from what it OBSERVED, never from the manifest's
        `affected_services` — a binding to a declared name would tie gate 3 to an
        intention, which is the thing gate 3 exists to be independent of."""
        observations = _started(_live())
        gate2 = SR.gate_stack_starts(observations, affected_services=("worker",))
        self.assertEqual(gate2.subjects, ("worker#4242",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
