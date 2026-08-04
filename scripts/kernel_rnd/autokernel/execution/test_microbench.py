"""Tests for the T1 paired-block microbench runner.

Every test here is STRUCTURAL: it fails if the guarantee it names is removed.
Where a guard could be satisfied by refusing everything, there is a
compliant-path control immediately beside it that fails if the guard becomes
unconditional — a guard with no compliant control is indistinguishable from a
module that always says no.

NO BENCHMARK IS RUN BY THIS FILE. Every `llama-bench` output it parses is a
verbatim copy of a real run on this host (`testdata/recorded_llama_bench_*.json`,
provenance and digests in `testdata/recorded_llama_bench_PROVENANCE.json`). The
one place a real process is spawned is `TestProcessDiscipline`, which runs
`/bin/sleep` — not inference, not under a claim, and terminated by pid to prove
the termination path actually terminates.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from .. import schemas
from ..evaluator import api, recipes, statistics
from . import microbench as M

TESTDATA = Path(__file__).resolve().parent / "testdata"
REPO_ROOT = Path(__file__).resolve().parents[4]

CANONICAL = TESTDATA / "recorded_llama_bench_cpu_decode_canonical.json"
FA_OFF = TESTDATA / "recorded_llama_bench_cpu_decode_fa_off.json"
T192 = TESTDATA / "recorded_llama_bench_cpu_decode_192t.json"
PROVENANCE = TESTDATA / "recorded_llama_bench_PROVENANCE.json"

RECIPE_ID = "t1b.llama_cpu.llama_bench_decode.v1"

#: Taken from the canonical fixture so the recipe and the recorded output agree.
FIXTURE_MODEL = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
FIXTURE_N_GEN = 128
FIXTURE_REPS = 10
FIXTURE_BUILD_COMMIT = "91745611f"

ANCHOR_COMMIT = "91745611f" + "0" * 31          # a 40-hex commit with the real prefix


def read_fixture(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def scaled_fixture(path: Path, *, factor: float, build_commit: str | None = None) -> str:
    """A DERIVED A/B arm: the real sample vector, scaled by a stated factor.

    This is the only place any number in these tests is not verbatim real
    output, and it is labelled as derived rather than shipped in `testdata/`
    where it could be mistaken for a recorded run. It exists because an A/B test
    needs two arms that differ, and the host is too contended tonight to measure
    a second one.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["samples_ts"] = [round(v * factor, 6) for v in row["samples_ts"]]
        row["avg_ts"] = round(sum(row["samples_ts"]) / len(row["samples_ts"]), 6)
        if build_commit is not None:
            row["build_commit"] = build_commit
    return json.dumps(rows)


# =============================================================================
# Bindings — temp directories shaped exactly as ToolBinding requires
# =============================================================================

class BindingFixture:
    """Two `llama-bench` bindings in a temp tree: a candidate and an anchor.

    They are real files with real content, so `integrity.sha256_file` produces a
    real digest and the receipt tests are not testing a mock.
    """

    def __init__(self, stack: "unittest.TestCase") -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="autokernel-microbench-test-")
        stack.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.candidate = self._make(root / "candidate", b"candidate-build-bytes")
        self.anchor = self._make(root / "anchor", b"anchor-build-bytes")
        self.model = root / "model.gguf"
        self.model.write_bytes(b"gguf")

    @staticmethod
    def _make(root: Path, payload: bytes) -> recipes.ToolBinding:
        bindir = root / "bin"
        bindir.mkdir(parents=True)
        binary = bindir / "llama-bench"
        binary.write_bytes(payload)
        binary.chmod(0o755)
        return recipes.ToolBinding(binary=str(binary), source_root=str(root),
                                   library_path=str(bindir))


def default_params(model: str = FIXTURE_MODEL) -> dict:
    return {"model": model, "n_gen": FIXTURE_N_GEN, "reps": FIXTURE_REPS,
            "output_format": "json"}


def build_command(binding: recipes.ToolBinding, *, arm: str = M.ARM_CANDIDATE,
                  params: dict | None = None) -> recipes.ConstructedCommand:
    return recipes.construct(RECIPE_ID, binding=binding, params=params or default_params(),
                             arm=arm, verify_inputs=False)


# =============================================================================
# Claims and host state — the two things that gate execution
# =============================================================================

class StubClaim:
    """A `HeldClaim` whose answer can change between attestations.

    A claim that caches its verdict would make the mid-run revocation check
    untestable AND useless, so this one re-evaluates on every call — which is
    also what the Protocol's docstring requires of a real implementation.
    """

    claim_id = "cpu_region.test"

    def __init__(self, outcomes: list | None = None,
                 outcome: str = schemas.PASS) -> None:
        self.outcomes = list(outcomes) if outcomes is not None else None
        self.default = outcome
        self.calls = 0

    def attest(self) -> M.ClaimAttestation:
        if self.outcomes:
            outcome = self.outcomes.pop(0)
        else:
            outcome = self.default
        self.calls += 1
        return M.ClaimAttestation(
            claim_id=self.claim_id, holder="pid:test", cpu_list="0-95",
            observed_at="2026-08-03T22:00:00+00:00",
            check=schemas.Check(outcome, () if outcome == schemas.PASS
                                else ("the region lock is not held by this process",)))


def fake_sysfs(tmp: Path, *, cpus: range, khz: int, min_khz: int = 400000,
               max_khz: int = 4510000, throttled: dict | None = None) -> Path:
    """A synthesised cpufreq tree, so the throttle guard is exercisable on a healthy host."""
    root = tmp / "sys-cpu"
    for cpu in cpus:
        d = root / f"cpu{cpu}" / "cpufreq"
        d.mkdir(parents=True, exist_ok=True)
        value = (throttled or {}).get(cpu, khz)
        (d / "scaling_cur_freq").write_text(f"{value}\n")
        (d / "cpuinfo_min_freq").write_text(f"{min_khz}\n")
        (d / "cpuinfo_max_freq").write_text(f"{max_khz}\n")
    return root


def fake_proc(tmp: Path, *, load1: float) -> Path:
    root = tmp / "proc"
    root.mkdir(parents=True, exist_ok=True)
    (root / "loadavg").write_text(f"{load1} {load1} {load1} 1/100 1234\n")
    return root


class HostStateStub:
    """A `read_host_state` replacement returning a scripted sequence of states."""

    def __init__(self, states: list) -> None:
        self.states = list(states)
        self.calls = 0

    def __call__(self, *, cpu_list: str, **kwargs) -> M.HostState:
        state = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        return replace(state, cpu_list=cpu_list)


def healthy_state(*, khz: int = 3500000, load1: float = 2.0) -> M.HostState:
    return M.HostState(
        observed_at="2026-08-03T22:00:00+00:00", cpu_list="0-95",
        khz_by_cpu=tuple((c, khz) for c in range(96)),
        driver_min_khz=400000, driver_max_khz=4510000, load1=load1, source="stub")


HEALTHY_POLICY = M.HostStatePolicy(nominal_khz=3500000)


# =============================================================================
# 1. Requirement 1 — paired blocks that cannot be mislabelled
# =============================================================================

class TestPairingIsStructural(unittest.TestCase):

    def _invocations(self, arms, *, plan, samples=(10.0, 11.0)):
        binding = BindingFixture(self)
        command = build_command(binding.candidate)
        receipt = M.build_receipt(command, env=M.assemble_env(command.env).env)
        claim = StubClaim().attest()
        return [
            M.Invocation(block_index=plan.block_index, position=i, arm=arm,
                         receipt=receipt,
                         spawn=M.SpawnResult(argv=("x",), returncode=0, stdout="",
                                             stderr_tail="", pid=None, duration_s=1.0),
                         row=None, samples=tuple(samples), claim=claim, checks=())
            for i, arm in enumerate(arms)
        ]

    def test_a_blocked_design_cannot_become_a_paired_block(self):
        """anchor,anchor,candidate,candidate is a BLOCKED design and must be refused."""
        plan = M.BlockPlan(block_index=0, order=statistics.ORDER_ANCHOR_FIRST, pairs=2,
                           unit_id="u0", stratum=api.STRATUM_SELECTION)
        blocked = [M.ARM_ANCHOR, M.ARM_ANCHOR, M.ARM_CANDIDATE, M.ARM_CANDIDATE]
        with self.assertRaises(M.PairingViolation) as ctx:
            M.assemble_block(plan, self._invocations(blocked, plan=plan))
        self.assertIn("plan requires", str(ctx.exception))

    def test_the_alternation_check_is_independent_of_the_plan(self):
        """Even a plan that ITSELF declares a blocked sequence cannot produce a block.

        `BlockPlan` derives its sequence, so this reaches past it by forging a
        plan object whose `arm_sequence` was overwritten after construction. The
        second check in `assemble_block` — strict alternation, evaluated without
        reference to the plan — is what catches it. Delete that loop and this
        test passes a blocked design.
        """
        plan = M.BlockPlan(block_index=0, order=statistics.ORDER_ANCHOR_FIRST, pairs=2,
                           unit_id="u0", stratum=api.STRATUM_SELECTION)
        blocked = (M.ARM_ANCHOR, M.ARM_ANCHOR, M.ARM_CANDIDATE, M.ARM_CANDIDATE)
        object.__setattr__(plan, "arm_sequence", blocked)
        with self.assertRaises(M.PairingViolation) as ctx:
            M.assemble_block(plan, self._invocations(list(blocked), plan=plan))
        self.assertIn("interleaved", str(ctx.exception))

    def test_block_plan_refuses_a_supplied_non_alternating_sequence(self):
        with self.assertRaises(M.PairingViolation):
            M.BlockPlan(block_index=0, order=statistics.ORDER_ANCHOR_FIRST, pairs=2,
                        unit_id="u0", stratum=api.STRATUM_SELECTION,
                        arm_sequence=(M.ARM_ANCHOR, M.ARM_ANCHOR,
                                      M.ARM_CANDIDATE, M.ARM_CANDIDATE))

    def test_the_recorded_order_comes_from_what_ran_not_from_what_was_declared(self):
        plan = M.BlockPlan(block_index=0, order=statistics.ORDER_ANCHOR_FIRST, pairs=1,
                           unit_id="u0", stratum=api.STRATUM_SELECTION)
        object.__setattr__(plan, "arm_sequence", (M.ARM_CANDIDATE, M.ARM_ANCHOR))
        with self.assertRaises(M.PairingViolation) as ctx:
            M.assemble_block(plan, self._invocations([M.ARM_CANDIDATE, M.ARM_ANCHOR],
                                                     plan=plan))
        self.assertIn("ran first", str(ctx.exception))

    def test_unequal_arms_are_not_a_paired_block(self):
        plan = M.BlockPlan(block_index=0, order=statistics.ORDER_ANCHOR_FIRST, pairs=1,
                           unit_id="u0", stratum=api.STRATUM_SELECTION)
        object.__setattr__(plan, "arm_sequence", (M.ARM_ANCHOR, M.ARM_CANDIDATE,
                                                  M.ARM_ANCHOR))
        with self.assertRaises(M.PairingViolation):
            M.assemble_block(plan, self._invocations(
                [M.ARM_ANCHOR, M.ARM_CANDIDATE, M.ARM_ANCHOR], plan=plan))

    def test_an_arm_with_no_samples_is_not_a_paired_block(self):
        plan = M.BlockPlan(block_index=0, order=statistics.ORDER_ANCHOR_FIRST, pairs=1,
                           unit_id="u0", stratum=api.STRATUM_SELECTION)
        invocations = self._invocations([M.ARM_ANCHOR, M.ARM_CANDIDATE], plan=plan)
        invocations[1] = replace(invocations[1], samples=())
        with self.assertRaises(M.PairingViolation) as ctx:
            M.assemble_block(plan, invocations)
        self.assertIn("no samples", str(ctx.exception))

    # -- compliant-path control ------------------------------------------

    def test_a_properly_interleaved_block_is_accepted(self):
        """The control. If this fails, the guard above has become unconditional."""
        plan = M.BlockPlan(block_index=3, order=statistics.ORDER_CANDIDATE_FIRST, pairs=2,
                           unit_id="u0", stratum=api.STRATUM_SELECTION)
        self.assertEqual(plan.arm_sequence,
                         (M.ARM_CANDIDATE, M.ARM_ANCHOR, M.ARM_CANDIDATE, M.ARM_ANCHOR))
        block = M.assemble_block(plan, self._invocations(list(plan.arm_sequence), plan=plan))
        self.assertIsInstance(block, statistics.PairedBlock)
        self.assertEqual(block.order, statistics.ORDER_CANDIDATE_FIRST)
        self.assertEqual(len(block.anchor_samples), 4)
        self.assertEqual(len(block.candidate_samples), 4)


class TestOrderComesFromTheCampaignSeed(unittest.TestCase):

    def test_the_runner_surface_has_no_order_parameter(self):
        """Requirement 1: order must not be declarable anywhere in the public API.

        If `order` or `arm_sequence` becomes a field of `MicrobenchPlan`, a
        caller can declare a schedule instead of deriving one, and
        `OrderSchedule.check_observed` would then be checking a run against its
        own assertion.
        """
        fields = set(M.MicrobenchPlan.__dataclass_fields__)
        self.assertNotIn("order", fields)
        self.assertNotIn("arm_sequence", fields)
        self.assertIn("campaign_seed", fields)

    def test_planned_orders_satisfy_the_reducers_own_order_control(self):
        schedule = statistics.OrderSchedule.derive(
            campaign_seed="seed-2026-08-03", candidate_id="cand-1", base_blocks=6)
        plans = M.plan_blocks(schedule, count=6, pairs=1, unit_ids=("u0", "u1"),
                              stratum=api.STRATUM_SELECTION)
        blocks = [
            statistics.PairedBlock(block_index=p.block_index, unit_id=p.unit_id,
                                   stratum=p.stratum, order=p.order,
                                   anchor_samples=(10.0,), candidate_samples=(11.0,))
            for p in plans
        ]
        self.assertEqual(schedule.check_observed(blocks).outcome, schemas.PASS)

    def test_a_schedule_derived_from_a_different_seed_does_not_validate(self):
        """The control's control: order control must be able to FAIL."""
        schedule = statistics.OrderSchedule.derive(
            campaign_seed="seed-A", candidate_id="cand-1", base_blocks=8)
        other = statistics.OrderSchedule.derive(
            campaign_seed="seed-B", candidate_id="cand-1", base_blocks=8)
        plans = M.plan_blocks(schedule, count=8, pairs=1, unit_ids=("u0",),
                              stratum=api.STRATUM_SELECTION)
        blocks = [
            statistics.PairedBlock(block_index=p.block_index, unit_id=p.unit_id,
                                   stratum=p.stratum, order=p.order,
                                   anchor_samples=(10.0,), candidate_samples=(11.0,))
            for p in plans
        ]
        self.assertEqual(other.check_observed(blocks).outcome, schemas.FAIL)

    def test_extension_blocks_do_not_change_the_base_schedule(self):
        schedule = statistics.OrderSchedule.derive(
            campaign_seed="seed-2026-08-03", candidate_id="cand-1", base_blocks=4)
        base = M.plan_blocks(schedule, count=4, pairs=1, unit_ids=("u0",),
                             stratum=api.STRATUM_SELECTION)
        extension = M.plan_blocks(schedule, count=2, pairs=1, unit_ids=("u0",),
                                  stratum=api.STRATUM_SELECTION,
                                  segment=statistics.SEGMENT_EXTENSION, extension_round=1)
        self.assertEqual([p.order for p in base], list(schedule.orders(4)))
        self.assertEqual([p.block_index for p in extension], [4, 5])
        # Extension pairs are fresh REVERSED-order pairs.
        self.assertEqual(extension[0].order,
                         statistics.OrderSchedule._flip(base[0].order))


# =============================================================================
# 2. Requirement 2 — the codified recipe, not defaults
# =============================================================================

class TestRecipeDiscipline(unittest.TestCase):

    def setUp(self):
        self.binding = BindingFixture(self)
        self.command = build_command(self.binding.candidate)
        self.env = M.assemble_env(self.command.env).env

    def test_argv_is_byte_identical_to_the_recipe_constructors_output(self):
        again = build_command(self.binding.candidate)
        self.assertEqual(self.command.argv, again.argv)

    def test_the_canonical_prefix_is_the_sourced_constant(self):
        prefix = list(recipes.CANONICAL_PREFIX)
        self.assertEqual(list(self.command.argv[:len(prefix)]), prefix)
        self.assertIn("--interleave=all", prefix)
        self.assertEqual(prefix[:2], ["taskset", "-c"])

    def test_flash_attention_is_on_where_llama_bench_defaults_it_off(self):
        argv = list(self.command.argv)
        self.assertEqual(argv[argv.index("-fa") + 1], "1")

    def test_the_full_omp_stack_is_in_the_executed_env(self):
        for key, value in recipes.CANONICAL_OMP_ENV.items():
            self.assertEqual(self.env.get(key), value, f"{key} missing or wrong")

    def test_discipline_check_catches_a_dropped_omp_variable(self):
        crippled = dict(self.env)
        crippled.pop("OMP_PROC_BIND")
        check = M.check_recipe_discipline(self.command, crippled)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("OMP stack incomplete" in r for r in check.reasons))

    def test_discipline_check_catches_flash_attention_turned_off(self):
        tampered = replace(self.command,
                           argv=tuple("0" if i and self.command.argv[i - 1] == "-fa" else t
                                      for i, t in enumerate(self.command.argv)))
        check = M.check_recipe_discipline(tampered, self.env)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("flash-attention ON" in r for r in check.reasons))

    def test_discipline_check_catches_flash_attention_absent_entirely(self):
        """`-fa` missing is a different branch from `-fa 0`, and llama-bench's own
        default is the missing case — so it is the one that actually happens."""
        stripped = replace(self.command, argv=tuple(
            t for i, t in enumerate(self.command.argv)
            if t != "-fa" and not (i and self.command.argv[i - 1] == "-fa")))
        self.assertNotIn("-fa", stripped.argv)
        check = M.check_recipe_discipline(stripped, self.env)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("defaults to -fa 0" in r for r in check.reasons))

    def test_discipline_check_catches_a_stripped_canonical_prefix(self):
        tampered = replace(self.command, argv=self.command.argv[5:])
        check = M.check_recipe_discipline(tampered, self.env)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("canonical prefix" in r for r in check.reasons))

    def test_discipline_check_catches_a_non_sample_bearing_output_format(self):
        command = build_command(self.binding.candidate,
                                params={**default_params(), "output_format": "md"})
        check = M.check_recipe_discipline(command, M.assemble_env(command.env).env)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("per-repetition sample vector" in r for r in check.reasons))

    # -- compliant-path control ------------------------------------------

    def test_the_unmodified_recipe_passes_discipline(self):
        check = M.check_recipe_discipline(self.command, self.env)
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)


class TestEnvironmentAssembly(unittest.TestCase):

    def setUp(self):
        self.binding = BindingFixture(self)
        self.command = build_command(self.binding.candidate)

    def test_ambient_variables_cannot_reach_the_measured_process(self):
        """An exported OMP_PROC_BIND=close must not shadow the recipe's `spread`.

        It would not appear anywhere in argv, so nothing downstream would notice.
        """
        poisoned = {"OMP_PROC_BIND": "close", "GGML_IQK": "0",
                    "LD_LIBRARY_PATH": "/somewhere/else", "PATH": "/usr/bin"}
        assembly = M.assemble_env(self.command.env, environ=poisoned)
        self.assertEqual(assembly.env["OMP_PROC_BIND"], "spread")
        self.assertEqual(assembly.env["GGML_IQK"], recipes.CANONICAL_OMP_ENV["GGML_IQK"])
        self.assertNotEqual(assembly.env["LD_LIBRARY_PATH"], "/somewhere/else")
        self.assertIn("OMP_PROC_BIND", assembly.dropped_ambient)

    def test_only_allowlisted_ambient_keys_survive(self):
        assembly = M.assemble_env(self.command.env,
                                  environ={"PATH": "/usr/bin", "SECRET": "x"})
        self.assertEqual(assembly.env.get("PATH"), "/usr/bin")
        self.assertNotIn("SECRET", assembly.env)

    def test_widening_the_allowlist_onto_a_recipe_key_is_refused(self):
        with self.assertRaises(ValueError):
            M.assemble_env(self.command.env, environ={"OMP_PLACES": "threads"},
                           base_keys=("OMP_PLACES",))

    def test_the_default_allowlist_is_disjoint_from_every_recipe_env_key(self):
        self.assertFalse(set(M.DEFAULT_BASE_ENV_KEYS) & set(self.command.env))


# =============================================================================
# 3. Requirement 3 — a receipt that can actually be verified
# =============================================================================

class TestReceiptVerification(unittest.TestCase):

    def setUp(self):
        self.binding = BindingFixture(self)
        self.command = build_command(self.binding.candidate)
        self.env = M.assemble_env(self.command.env).env
        self.receipt = M.build_receipt(self.command, env=self.env)

    def test_the_standing_gap_is_real_and_is_what_this_receipt_closes(self):
        """`api.check_preconditions` passes precondition 6 on PRESENCE alone.

        A receipt copied from a legitimate run and stapled to a hand-typed argv
        satisfies the gate today. This test pins the gap so the follow-up that
        closes it has something to turn red.
        """
        forged = api.RecipeReceipt(constructor_id="anything",
                                   constructor_sha256="a" * 64, argv_sha256="b" * 64)
        self.assertIsNotNone(forged)
        self.assertEqual(len(M.RECEIPT_VERIFICATION_REQUIREMENTS), 5)
        self.assertTrue(any("RESOLVED ARGV" in r
                            for r in M.RECEIPT_VERIFICATION_REQUIREMENTS))

    def test_a_single_mutated_argv_token_fails_verification(self):
        tampered = list(self.receipt.argv)
        tampered[tampered.index("-fa") + 1] = "0"
        check = M.verify_receipt(self.receipt, argv=tampered, env=self.env)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("argv does not match" in r for r in check.reasons))

    def test_a_consistently_rehashed_forgery_still_fails(self):
        """Re-hashing the edited argv is not enough: the tokens are compared too."""
        tampered = list(self.receipt.argv)
        tampered[tampered.index("-fa") + 1] = "0"
        rehashed = replace(
            self.receipt, argv_sha256=M._argv_hash(
                recipe_id=self.receipt.recipe_id, registry_id=self.receipt.registry_id,
                arm=self.receipt.arm, argv=tampered, env=self.receipt.recipe_env))
        check = M.verify_receipt(rehashed, argv=tampered, env=self.env)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("argv does not match" in r for r in check.reasons))

    def test_a_mutated_env_fails_verification(self):
        crippled = dict(self.env)
        crippled["OMP_PROC_BIND"] = "close"
        check = M.verify_receipt(self.receipt, argv=self.receipt.argv, env=crippled)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("env does not match" in r for r in check.reasons))

    def test_a_dropped_env_variable_fails_verification(self):
        crippled = dict(self.env)
        crippled.pop("OMP_WAIT_POLICY")
        check = M.verify_receipt(self.receipt, argv=self.receipt.argv, env=crippled)
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_a_swapped_binary_fails_verification(self):
        anchor_digest = hashlib.sha256(b"anchor-build-bytes").hexdigest()
        check = M.verify_receipt(self.receipt, argv=self.receipt.argv, env=self.env,
                                 binary_sha256=anchor_digest)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("is not the binary" in r for r in check.reasons))

    def test_the_receipt_digests_the_binary_that_would_run(self):
        self.assertEqual(self.receipt.binary_sha256,
                         hashlib.sha256(b"candidate-build-bytes").hexdigest())
        self.assertEqual(self.receipt.binary_size, len(b"candidate-build-bytes"))

    def test_independent_reconstruction_rebuilds_the_same_argv(self):
        check = M.verify_receipt(self.receipt, argv=self.receipt.argv, env=self.env,
                                 reconstruct=True)
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_independent_reconstruction_rejects_an_argv_the_recipe_cannot_emit(self):
        """The only check that trusts nothing the receipt says about itself."""
        hand_typed = list(self.receipt.argv) + ["--some-flag-no-recipe-emits"]
        rehashed = replace(
            self.receipt, argv=tuple(hand_typed),
            argv_sha256=M._argv_hash(
                recipe_id=self.receipt.recipe_id, registry_id=self.receipt.registry_id,
                arm=self.receipt.arm, argv=hand_typed, env=self.receipt.recipe_env))
        check = M.verify_receipt(rehashed, argv=hand_typed, env=self.env, reconstruct=True)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("independent reconstruction" in r for r in check.reasons))

    def test_the_recipe_env_must_survive_assembly_unchanged(self):
        """The assembly step must not be an unaudited place to alter a measurement."""
        altered = dict(self.env)
        altered["GGML_IQK"] = "0"
        check = M.verify_receipt(self.receipt, argv=self.receipt.argv, env=altered)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("alters recipe-declared variables" in r
                            for r in check.reasons))

    def test_the_receipt_keeps_both_environments(self):
        self.assertEqual(self.receipt.recipe_env, dict(self.command.env))
        self.assertTrue(set(self.receipt.recipe_env) < set(self.receipt.env))
        for key, value in self.receipt.recipe_env.items():
            self.assertEqual(self.receipt.env[key], value)

    def test_the_receipt_downgrades_to_the_three_field_api_form(self):
        api_receipt = self.receipt.recipe_receipt
        self.assertIsInstance(api_receipt, api.RecipeReceipt)
        self.assertEqual(api_receipt.argv_sha256, self.command.receipt.argv_sha256)
        self.assertEqual(api_receipt.constructor_id, self.command.receipt.constructor_id)

    def test_the_receipt_argv_hash_agrees_with_the_recipe_constructors_own(self):
        """Two independent computations of the same preimage must not drift."""
        self.assertEqual(
            M._argv_hash(recipe_id=self.command.recipe_id,
                         registry_id=self.command.registry_id, arm=self.command.arm,
                         argv=self.command.argv, env=self.command.env),
            self.command.receipt.argv_sha256)

    def test_the_receipt_is_canonical_json_able(self):
        schemas.canonical_json(self.receipt.to_dict())

    # -- compliant-path control ------------------------------------------

    def test_the_unmodified_receipt_verifies(self):
        check = M.verify_receipt(self.receipt, argv=self.receipt.argv, env=self.env,
                                 binary_sha256=self.receipt.binary_sha256)
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)


# =============================================================================
# Output parsing — against REAL recorded llama-bench output
# =============================================================================

class TestFixturesAreRealAndUnmodified(unittest.TestCase):
    """A fixture edited to make a test pass must fail this test instead."""

    def test_every_fixture_matches_its_recorded_digest(self):
        manifest = json.loads(PROVENANCE.read_text())
        self.assertTrue(manifest["files"])
        for name, entry in manifest["files"].items():
            digest = hashlib.sha256((TESTDATA / name).read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], f"{name} was modified")

    def test_every_fixture_is_byte_identical_to_its_source_in_this_repo(self):
        manifest = json.loads(PROVENANCE.read_text())
        checked = 0
        for name, entry in manifest["files"].items():
            source = REPO_ROOT / entry["source_path"]
            if not source.exists():
                continue
            self.assertEqual((TESTDATA / name).read_bytes(), source.read_bytes(),
                             f"{name} has diverged from {entry['source_path']}")
            checked += 1
        self.assertGreater(checked, 0, "no fixture source was resolvable in this repo; "
                                       "the provenance claim is unverifiable")


class TestLlamaBenchParsing(unittest.TestCase):

    def test_parses_the_real_canonical_run(self):
        rows = M.parse_llama_bench_json(read_fixture(CANONICAL))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.n_threads, 96)
        self.assertTrue(row.flash_attn)
        self.assertFalse(row.use_mmap)
        self.assertEqual(row.n_gen, 128)
        self.assertEqual(row.n_prompt, 0)
        self.assertEqual(len(row.samples_ts), 10)
        self.assertEqual(row.samples_ts[0], 12.4391)
        self.assertEqual(row.build_commit, FIXTURE_BUILD_COMMIT)

    def test_the_reduction_uses_raw_samples_not_the_tools_average(self):
        """`avg_ts` is retained for cross-checking and never reduced from."""
        row = M.parse_llama_bench_json(read_fixture(CANONICAL))[0]
        self.assertEqual(row.metric_samples, row.samples_ts)
        recomputed = sum(row.samples_ts) / len(row.samples_ts)
        self.assertAlmostEqual(recomputed, row.avg_ts, places=3)

    def test_flash_attn_reads_both_the_bool_and_int_spellings(self):
        self.assertTrue(M.parse_llama_bench_json(read_fixture(CANONICAL))[0].flash_attn)
        self.assertFalse(M.parse_llama_bench_json(read_fixture(FA_OFF))[0].flash_attn)

    def test_markdown_output_is_refused(self):
        md = ("| model | size | test |   t/s |\n"
              "| ----- | ---: | ---: | ----: |\n"
              "| qwen  |  35B | tg128 | 12.44 |\n")
        with self.assertRaises(M.BenchOutputError) as ctx:
            M.parse_llama_bench_json(md)
        self.assertIn("per-repetition samples", str(ctx.exception))

    def test_a_row_without_samples_is_refused(self):
        rows = json.loads(read_fixture(CANONICAL))
        rows[0].pop("samples_ts")
        with self.assertRaises(M.BenchOutputError) as ctx:
            M.parse_llama_bench_json(json.dumps(rows))
        self.assertIn("samples_ts", str(ctx.exception))

    def test_a_row_missing_a_required_field_refuses_rather_than_raising_keyerror(self):
        """A malformed row must come back as `BenchOutputError`, not a bare KeyError.

        The runner catches `BenchOutputError` and journals the reason; a KeyError
        escapes the block loop and loses it.
        """
        for missing in ("avg_ts", "n_threads", "build_commit"):
            with self.subTest(missing=missing):
                rows = json.loads(read_fixture(CANONICAL))
                rows[0].pop(missing)
                with self.assertRaises(M.BenchOutputError) as ctx:
                    M.parse_llama_bench_json(json.dumps(rows))
                self.assertIn(missing, str(ctx.exception))

    def test_an_empty_sample_vector_is_refused(self):
        rows = json.loads(read_fixture(CANONICAL))
        rows[0]["samples_ts"] = []
        with self.assertRaises(M.BenchOutputError):
            M.parse_llama_bench_json(json.dumps(rows))

    def test_a_zero_throughput_sample_is_refused(self):
        rows = json.loads(read_fixture(CANONICAL))
        rows[0]["samples_ts"][3] = 0.0
        with self.assertRaises(M.BenchOutputError) as ctx:
            M.parse_llama_bench_json(json.dumps(rows))
        self.assertIn("not positive", str(ctx.exception))

    def test_empty_stdout_is_refused(self):
        with self.assertRaises(M.BenchOutputError):
            M.parse_llama_bench_json("   \n")

    def test_a_json_object_instead_of_an_array_is_refused(self):
        with self.assertRaises(M.BenchOutputError):
            M.parse_llama_bench_json('{"avg_ts": 12.4}')


class TestOutputIsCheckedAgainstTheRecipe(unittest.TestCase):
    """The flag that is present in argv and did not take effect."""

    def setUp(self):
        self.binding = BindingFixture(self)
        self.command = build_command(self.binding.candidate)
        self.expect = M.LlamaBenchExpectation.from_command(self.command)

    def test_the_expectation_is_read_out_of_argv(self):
        self.assertEqual(self.expect.n_threads, 96)
        self.assertTrue(self.expect.flash_attn)
        self.assertEqual(self.expect.n_gen, FIXTURE_N_GEN)
        self.assertEqual(self.expect.reps, FIXTURE_REPS)
        self.assertEqual(self.expect.model_filename, FIXTURE_MODEL)

    def test_real_output_with_flash_attention_off_is_refused(self):
        """Real recorded output, flash_attn=false, against an `-fa 1` recipe."""
        row = M.parse_llama_bench_json(read_fixture(FA_OFF))[0]
        check = self.expect.check_row(row)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("did not take effect" in r for r in check.reasons))

    def test_real_output_at_the_wrong_thread_count_is_refused(self):
        row = M.parse_llama_bench_json(read_fixture(T192))[0]
        check = self.expect.check_row(row)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("n_threads" in r for r in check.reasons))

    def test_a_short_sample_vector_is_refused(self):
        rows = json.loads(read_fixture(CANONICAL))
        rows[0]["samples_ts"] = rows[0]["samples_ts"][:4]
        row = M.parse_llama_bench_json(json.dumps(rows))[0]
        check = self.expect.check_row(row)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("repetitions were dropped" in r for r in check.reasons))

    def test_a_different_prompt_decode_point_is_a_different_cell(self):
        rows = json.loads(read_fixture(CANONICAL))
        rows[0]["n_gen"] = 64
        row = M.parse_llama_bench_json(json.dumps(rows))[0]
        check = self.expect.check_row(row)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("different cell" in r for r in check.reasons))

    def test_an_anchor_that_reports_a_different_build_commit_is_refused(self):
        """The anchor's immutability, checked in the OUTPUT rather than assumed."""
        anchored = M.LlamaBenchExpectation.from_command(
            self.command, expected_build_commit="deadbee" + "f" * 33)
        row = M.parse_llama_bench_json(read_fixture(CANONICAL))[0]
        check = anchored.check_row(row)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("the anchor is not the anchor" in r for r in check.reasons))

    # -- compliant-path controls -----------------------------------------

    def test_the_matching_real_row_passes(self):
        row = M.parse_llama_bench_json(read_fixture(CANONICAL))[0]
        self.assertEqual(self.expect.check_row(row).outcome, schemas.PASS)

    def test_an_anchor_whose_build_commit_prefix_matches_passes(self):
        anchored = M.LlamaBenchExpectation.from_command(
            self.command, expected_build_commit=ANCHOR_COMMIT)
        row = M.parse_llama_bench_json(read_fixture(CANONICAL))[0]
        self.assertEqual(anchored.check_row(row).outcome, schemas.PASS)


# =============================================================================
# 5. Requirement 5 — honest failure
# =============================================================================

class TestHostStateGuards(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_reads_per_cpu_frequency_from_sysfs(self):
        sysfs = fake_sysfs(self.root, cpus=range(8), khz=3500000)
        proc = fake_proc(self.root, load1=1.5)
        state = M.read_host_state(cpu_list="0-7", sysfs_root=sysfs, proc_root=proc)
        self.assertEqual(len(state.khz_by_cpu), 8)
        self.assertEqual(state.min_khz, 3500000)
        self.assertEqual(state.load1, 1.5)

    def test_a_single_throttled_core_is_visible_in_the_minimum(self):
        """A throttle on one CCD is invisible in a mean and obvious in a minimum."""
        sysfs = fake_sysfs(self.root, cpus=range(8), khz=3500000, throttled={5: 1400000})
        proc = fake_proc(self.root, load1=1.0)
        state = M.read_host_state(cpu_list="0-7", sysfs_root=sysfs, proc_root=proc)
        self.assertEqual(state.min_khz, 1400000)
        check = M.HostStatePolicy(nominal_khz=3500000).check_frequency(state)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("throttled host" in r for r in check.reasons))

    def test_the_multi_day_sixty_percent_throttle_is_caught(self):
        """The actual scar: this host sat at ~40% of clock for days."""
        sysfs = fake_sysfs(self.root, cpus=range(8), khz=1400000)
        proc = fake_proc(self.root, load1=0.5)
        state = M.read_host_state(cpu_list="0-7", sysfs_root=sysfs, proc_root=proc)
        check = M.HostStatePolicy(nominal_khz=3500000).check_frequency(state)
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_a_clock_pinned_at_the_driver_minimum_fails_without_a_nominal(self):
        """The one throttle shape that needs no operator-supplied reference."""
        sysfs = fake_sysfs(self.root, cpus=range(4), khz=400000, min_khz=400000)
        proc = fake_proc(self.root, load1=0.1)
        state = M.read_host_state(cpu_list="0-3", sysfs_root=sysfs, proc_root=proc)
        check = M.HostStatePolicy().check_frequency(state)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("driver's own minimum" in r for r in check.reasons))

    def test_an_unreadable_frequency_is_not_a_passing_frequency(self):
        proc = fake_proc(self.root, load1=0.1)
        state = M.read_host_state(cpu_list="0-3", sysfs_root=self.root / "absent",
                                  proc_root=proc)
        check = M.HostStatePolicy(nominal_khz=3500000).check_frequency(state)
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_no_nominal_reference_is_could_not_check_not_pass(self):
        sysfs = fake_sysfs(self.root, cpus=range(4), khz=3500000)
        proc = fake_proc(self.root, load1=0.1)
        state = M.read_host_state(cpu_list="0-3", sysfs_root=sysfs, proc_root=proc)
        check = M.HostStatePolicy().check_frequency(state)
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        self.assertTrue(any("boost ceiling" in r for r in check.reasons))

    def test_a_contended_host_fails_the_load_check(self):
        """Tonight's actual condition: load ~67-97 on a 96-core box."""
        sysfs = fake_sysfs(self.root, cpus=range(96), khz=3500000)
        proc = fake_proc(self.root, load1=67.0)
        state = M.read_host_state(cpu_list="0-95", sysfs_root=sysfs, proc_root=proc)
        check = M.HostStatePolicy(nominal_khz=3500000).check_load(state, cpu_count=96)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("garbage data" in r for r in check.reasons))

    def test_an_unreadable_loadavg_is_could_not_check(self):
        sysfs = fake_sysfs(self.root, cpus=range(4), khz=3500000)
        state = M.read_host_state(cpu_list="0-3", sysfs_root=sysfs,
                                  proc_root=self.root / "absent")
        check = M.HostStatePolicy(nominal_khz=3500000).check_load(state, cpu_count=4)
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_cpu_list_grammar_matches_tasksets(self):
        self.assertEqual(M._parse_cpu_list("0-3"), (0, 1, 2, 3))
        self.assertEqual(M._parse_cpu_list("0,2,4"), (0, 2, 4))
        self.assertEqual(M._parse_cpu_list("184-187"), (184, 185, 186, 187))
        with self.assertRaises(ValueError):
            M._parse_cpu_list("5-1")

    # -- compliant-path controls -----------------------------------------

    def test_a_healthy_quiet_host_passes_both_checks(self):
        sysfs = fake_sysfs(self.root, cpus=range(96), khz=3500000)
        proc = fake_proc(self.root, load1=2.0)
        state = M.read_host_state(cpu_list="0-95", sysfs_root=sysfs, proc_root=proc)
        policy = M.HostStatePolicy(nominal_khz=3500000)
        self.assertEqual(policy.check_frequency(state).outcome, schemas.PASS)
        self.assertEqual(policy.check_load(state, cpu_count=96).outcome, schemas.PASS)


# =============================================================================
# The runner, end to end, on recorded output
# =============================================================================

def anchor_identity(binding: BindingFixture, *, binary_sha256: str | None = None,
                    source_commit: str = ANCHOR_COMMIT) -> api.AnchorIdentity:
    """The anchor triple, with the digest of the anchor binary that will REALLY run.

    Deliberately derived from the fixture rather than stapled on as `"a" * 64`.
    The runner compares `AnchorIdentity.binary_sha256` against the digest it takes
    of the binary it is about to execute, so a placeholder here would make every
    end-to-end test in this file a test of a run the runner refuses.
    """
    return api.AnchorIdentity(
        source_commit=source_commit,
        binary_sha256=binary_sha256 or hashlib.sha256(
            Path(binding.anchor.binary).read_bytes()).hexdigest(),
        linkage_sha256="b" * 64)


def make_plan(binding: BindingFixture, *, blocks: int = 4, pairs: int = 1,
              anchor: api.AnchorIdentity | None = None,
              **kwargs) -> M.MicrobenchPlan:
    return M.MicrobenchPlan(
        recipe_id=RECIPE_ID, candidate_id="cand-alpha",
        campaign_seed="campaign-seed-2026-08-03",
        candidate_binding=binding.candidate, anchor_binding=binding.anchor,
        anchor=anchor or anchor_identity(binding),
        params=default_params(), base_blocks=blocks, pairs_per_block=pairs,
        unit_ids=("unit-0",), **kwargs)


def arm_aware_spawner(*, candidate_stdout: str, anchor_stdout: str) -> M.RecordedSpawner:
    """Routes by which binding's path is in argv — the same way the runner does."""
    return M.RecordedSpawner({M.ARM_CANDIDATE: candidate_stdout,
                              M.ARM_ANCHOR: anchor_stdout})


class TestRunnerEndToEnd(unittest.TestCase):

    def setUp(self):
        self.binding = BindingFixture(self)
        self.anchor_out = read_fixture(CANONICAL)
        self.candidate_out = scaled_fixture(CANONICAL, factor=1.08,
                                            build_commit="cafe12345")

    def _runner(self, *, claim=None, states=None, policy=HEALTHY_POLICY,
                spawner=None) -> M.MicrobenchRunner:
        return M.MicrobenchRunner(
            claim=claim or StubClaim(), policy=policy,
            spawner=spawner or arm_aware_spawner(candidate_stdout=self.candidate_out,
                                                 anchor_stdout=self.anchor_out),
            host_state=HostStateStub(states or [healthy_state()]))

    def test_a_healthy_run_completes_and_yields_paired_blocks(self):
        run = self._runner().run(make_plan(self.binding, blocks=4))
        self.assertTrue(run.complete, run.refusals)
        blocks = run.paired_blocks()
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            self.assertEqual(len(block.anchor_samples), FIXTURE_REPS)
            self.assertEqual(len(block.candidate_samples), FIXTURE_REPS)

    def test_the_emitted_blocks_satisfy_the_reducers_order_control(self):
        plan = make_plan(self.binding, blocks=6)
        run = self._runner().run(plan)
        self.assertEqual(plan.schedule().check_observed(run.paired_blocks()).outcome,
                         schemas.PASS)

    def test_the_arms_actually_alternated_in_the_spawn_log(self):
        spawner = arm_aware_spawner(candidate_stdout=self.candidate_out,
                                    anchor_stdout=self.anchor_out)
        run = self._runner(spawner=spawner).run(make_plan(self.binding, blocks=3, pairs=2))
        self.assertTrue(run.complete, run.refusals)
        arms = [call["arm"] for call in spawner.calls]
        self.assertEqual(len(arms), 3 * 2 * 2)
        for i in range(1, len(arms)):
            if i % 4:                                   # within a block
                self.assertNotEqual(arms[i], arms[i - 1], f"blocked design at {i}")

    def test_every_spawn_received_the_recipe_argv_and_the_full_omp_stack(self):
        spawner = arm_aware_spawner(candidate_stdout=self.candidate_out,
                                    anchor_stdout=self.anchor_out)
        self._runner(spawner=spawner).run(make_plan(self.binding, blocks=2))
        expected_prefix = list(recipes.CANONICAL_PREFIX)
        self.assertTrue(spawner.calls)
        for call in spawner.calls:
            self.assertEqual(list(call["argv"][:len(expected_prefix)]), expected_prefix)
            self.assertEqual(call["argv"][call["argv"].index("-fa") + 1], "1")
            for key, value in recipes.CANONICAL_OMP_ENV.items():
                self.assertEqual(call["env"][key], value)

    def test_the_raw_vector_carries_every_sample_and_the_scope_denominator(self):
        run = self._runner().run(make_plan(self.binding, blocks=2))
        vector = run.raw_vector()
        self.assertIn("scope_denominator", vector)
        self.assertEqual(vector["scope_denominator"]["cores"], 96)
        self.assertTrue(vector["scope_render"])
        emitted = [s for b in vector["blocks"] for i in b["invocations"]
                   for s in i["samples"]]
        self.assertEqual(len(emitted), 2 * 2 * FIXTURE_REPS)
        schemas.canonical_json(vector)

    def test_the_raw_vector_is_not_merely_a_summary(self):
        """A summary would carry a median per block; this carries every repetition."""
        run = self._runner().run(make_plan(self.binding, blocks=1))
        invocation = run.raw_vector()["blocks"][0]["invocations"][0]
        self.assertEqual(len(invocation["samples"]), FIXTURE_REPS)
        self.assertEqual(invocation["row"]["samples_ts"], list(invocation["samples"]))

    def test_the_run_records_which_argv_ran_and_it_verifies(self):
        spawner = arm_aware_spawner(candidate_stdout=self.candidate_out,
                                    anchor_stdout=self.anchor_out)
        run = self._runner(spawner=spawner).run(make_plan(self.binding, blocks=1))
        candidate_call = next(c for c in spawner.calls if c["arm"] == M.ARM_CANDIDATE)
        check = M.verify_receipt(run.candidate_receipt, argv=candidate_call["argv"],
                                 env=candidate_call["env"])
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_two_arms_get_different_receipts(self):
        run = self._runner().run(make_plan(self.binding, blocks=1))
        self.assertNotEqual(run.candidate_receipt.binary_sha256,
                            run.anchor_receipt.binary_sha256)
        self.assertNotEqual(run.candidate_receipt.argv_sha256,
                            run.anchor_receipt.argv_sha256)


class TestRunnerRefusesRatherThanEmitting(unittest.TestCase):

    def setUp(self):
        self.binding = BindingFixture(self)
        self.anchor_out = read_fixture(CANONICAL)
        self.candidate_out = scaled_fixture(CANONICAL, factor=1.08,
                                            build_commit="cafe12345")

    def _spawner(self, **kw):
        return arm_aware_spawner(candidate_stdout=self.candidate_out,
                                 anchor_stdout=self.anchor_out, **kw)

    def test_no_claim_means_no_runner_at_all(self):
        with self.assertRaises(M.ClaimNotHeld):
            M.MicrobenchRunner(claim=None, spawner=self._spawner())

    def test_an_unheld_claim_spawns_absolutely_nothing(self):
        """Denial 8. The assertion that matters is `spawner.calls == []`."""
        spawner = self._spawner()
        runner = M.MicrobenchRunner(claim=StubClaim(outcome=schemas.FAIL),
                                    spawner=spawner, policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=2))
        self.assertEqual(spawner.calls, [])
        self.assertFalse(run.complete)
        self.assertTrue(any("claim was not held" in r for r in run.refusals))
        with self.assertRaises(M.RunRefused):
            run.paired_blocks()

    def test_a_could_not_check_claim_is_not_a_held_claim(self):
        spawner = self._spawner()
        runner = M.MicrobenchRunner(claim=StubClaim(outcome=schemas.COULD_NOT_CHECK),
                                    spawner=spawner, policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=1))
        self.assertEqual(spawner.calls, [])
        self.assertFalse(run.complete)

    def test_a_claim_revoked_mid_run_stops_at_the_next_invocation(self):
        """The claim is re-attested before EVERY spawn, not once at the top."""
        spawner = self._spawner()
        claim = StubClaim(outcomes=[schemas.PASS, schemas.PASS, schemas.PASS,
                                    schemas.FAIL, schemas.FAIL, schemas.FAIL])
        runner = M.MicrobenchRunner(claim=claim, spawner=spawner, policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=4))
        self.assertEqual(len(spawner.calls), 3)
        self.assertFalse(run.complete)
        self.assertTrue(any("claim was not held" in r for r in run.refusals))
        with self.assertRaises(M.RunRefused):
            run.paired_blocks()

    def test_a_claim_that_does_not_cover_the_argv_footprint_spawns_nothing(self):
        """Precondition 1: the claim must cover the EXACT footprint measured.

        A claim on cores 0-47 while the argv pins `taskset -c 0-95` leaves half
        the measured machine unprotected, and every journal field still reads
        like a claimed run.
        """
        spawner = self._spawner()
        narrow = StubClaim()
        narrow.attest = lambda: M.ClaimAttestation(          # type: ignore[assignment]
            claim_id="cpu_region.narrow", holder="pid:test", cpu_list="0-47",
            observed_at="2026-08-03T22:00:00+00:00", check=schemas.Check(schemas.PASS))
        runner = M.MicrobenchRunner(claim=narrow, spawner=spawner, policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=2))
        self.assertEqual(spawner.calls, [])
        self.assertFalse(run.complete)
        self.assertTrue(any("outside the claim" in r.lower() or "OUTSIDE the" in r
                            for r in run.refusals), run.refusals)

    def test_a_claim_wider_than_the_footprint_is_accepted(self):
        """Compliant control: covering MORE than the footprint is still covering it."""
        spawner = self._spawner()
        wide = StubClaim()
        wide.attest = lambda: M.ClaimAttestation(            # type: ignore[assignment]
            claim_id="cpu_region.wide", holder="pid:test", cpu_list="0-191",
            observed_at="2026-08-03T22:00:00+00:00", check=schemas.Check(schemas.PASS))
        runner = M.MicrobenchRunner(claim=wide, spawner=spawner, policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=2))
        self.assertTrue(run.complete, run.refusals)

    def test_a_contended_host_refuses_before_spawning(self):
        spawner = self._spawner()
        runner = M.MicrobenchRunner(
            claim=StubClaim(), spawner=spawner, policy=HEALTHY_POLICY,
            host_state=HostStateStub([healthy_state(load1=67.0)]))
        run = runner.run(make_plan(self.binding, blocks=2))
        self.assertEqual(spawner.calls, [])
        self.assertTrue(any("contention" in r for r in run.refusals))
        with self.assertRaises(M.RunRefused):
            run.paired_blocks()

    def test_a_throttled_host_refuses_before_spawning(self):
        spawner = self._spawner()
        throttled = replace(healthy_state(),
                            khz_by_cpu=tuple((c, 1400000) for c in range(96)))
        runner = M.MicrobenchRunner(
            claim=StubClaim(), spawner=spawner, policy=HEALTHY_POLICY,
            host_state=HostStateStub([throttled]))
        run = runner.run(make_plan(self.binding, blocks=2))
        self.assertEqual(spawner.calls, [])
        self.assertTrue(any("frequency" in r for r in run.refusals))

    def test_a_throttle_that_develops_mid_run_voids_the_block_it_developed_in(self):
        spawner = self._spawner()
        throttled = replace(healthy_state(),
                            khz_by_cpu=tuple((c, 1200000) for c in range(96)))
        runner = M.MicrobenchRunner(
            claim=StubClaim(), spawner=spawner, policy=HEALTHY_POLICY,
            host_state=HostStateStub([healthy_state(), healthy_state(), throttled]))
        run = runner.run(make_plan(self.binding, blocks=3))
        self.assertFalse(run.complete)
        self.assertTrue(any("block close" in r or "frequency" in r for r in run.refusals))
        with self.assertRaises(M.RunRefused):
            run.paired_blocks()

    def test_a_nonzero_exit_refuses_and_emits_no_samples(self):
        spawner = M.RecordedSpawner({}, default=M.SpawnResult(
            argv=("x",), returncode=1, stdout="", stderr_tail="ggml assert failed",
            pid=4242, duration_s=0.4))
        runner = M.MicrobenchRunner(claim=StubClaim(), spawner=spawner,
                                    policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=2))
        self.assertFalse(run.complete)
        self.assertTrue(any("exit 1" in r for r in run.refusals))
        self.assertEqual([s for b in run.blocks for i in b.invocations for s in i.samples],
                         [])

    def test_a_timeout_refuses_rather_than_reporting_a_short_run(self):
        spawner = M.RecordedSpawner({}, default=M.SpawnResult(
            argv=("x",), returncode=-15, stdout="", stderr_tail="", pid=4242,
            duration_s=1800.0, timed_out=True, terminated_by_runner=True))
        runner = M.MicrobenchRunner(claim=StubClaim(), spawner=spawner,
                                    policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=1))
        self.assertFalse(run.complete)
        self.assertTrue(any("terminated by this runner" in r for r in run.refusals))

    def test_output_disagreeing_with_the_recipe_refuses_the_run(self):
        """Real fa-off output arriving from an `-fa 1` argv."""
        spawner = arm_aware_spawner(candidate_stdout=read_fixture(FA_OFF),
                                    anchor_stdout=self.anchor_out)
        runner = M.MicrobenchRunner(claim=StubClaim(), spawner=spawner,
                                    policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=2))
        self.assertFalse(run.complete)
        self.assertTrue(any("did not take effect" in r for r in run.refusals))

    def test_a_multi_row_sweep_is_refused_as_a_shared_record(self):
        rows = json.loads(read_fixture(CANONICAL)) * 2
        spawner = arm_aware_spawner(candidate_stdout=json.dumps(rows),
                                    anchor_stdout=self.anchor_out)
        runner = M.MicrobenchRunner(claim=StubClaim(), spawner=spawner,
                                    policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=1))
        self.assertFalse(run.complete)
        self.assertTrue(any("different cell" in r for r in run.refusals))

    def test_a_refused_run_still_retains_its_raw_vector_and_reasons(self):
        """A failure that is not durable is indistinguishable from one that never ran."""
        spawner = M.RecordedSpawner({}, default=M.SpawnResult(
            argv=("x",), returncode=1, stdout="", stderr_tail="boom", pid=1,
            duration_s=0.1))
        runner = M.MicrobenchRunner(claim=StubClaim(), spawner=spawner,
                                    policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=2))
        vector = run.raw_vector()
        self.assertFalse(vector["complete"])
        self.assertTrue(vector["refusals"])
        self.assertTrue(vector["blocks"])
        schemas.canonical_json(vector)

    def test_a_short_run_does_not_emit_a_number(self):
        spawner = self._spawner()
        claim = StubClaim(outcomes=[schemas.PASS] * 4 + [schemas.FAIL] * 20)
        runner = M.MicrobenchRunner(claim=claim, spawner=spawner, policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=8))
        self.assertLess(len(run.blocks), 8)
        with self.assertRaises(M.RunRefused) as ctx:
            run.paired_blocks()
        self.assertIn("blocks completed", str(ctx.exception))

    def test_a_short_run_says_it_was_short_rather_than_merely_failing_to_complete(self):
        """A refused run must carry its REASON, not just an unset `complete` flag.

        `complete` already goes False on a short block list, so a run that loses
        the shortfall reason still refuses — and refuses without being able to
        say why, which is a failure that is not durable.
        """
        runner = M.MicrobenchRunner(claim=StubClaim(), spawner=self._spawner(),
                                    policy=HEALTHY_POLICY,
                                    host_state=HostStateStub([healthy_state()]))
        plan = make_plan(self.binding, blocks=5)
        run = runner._finish(plan, "2026-08-03T22:00:00+00:00", [], [], [],
                             api.ScopeDenominator(machine_subset="full", numa_nodes=(),
                                                  devices=(), cores=96), [], {})
        self.assertFalse(run.complete)
        self.assertTrue(any("0/5 paired blocks completed" in r for r in run.refusals),
                        run.refusals)

    # -- compliant-path control ------------------------------------------

    def test_the_healthy_run_is_not_refused(self):
        runner = M.MicrobenchRunner(
            claim=StubClaim(), spawner=self._spawner(), policy=HEALTHY_POLICY,
            host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(self.binding, blocks=2))
        self.assertEqual(run.refusals, ())
        self.assertTrue(run.complete)
        self.assertEqual(len(run.paired_blocks()), 2)


# =============================================================================
# The seam into the reducer — "produce exactly what the reducer consumes"
# =============================================================================

class TestTheReducerAcceptsWhatTheRunnerEmits(unittest.TestCase):

    def setUp(self):
        self.binding = BindingFixture(self)

    def _run(self, factor: float) -> M.MicrobenchRun:
        runner = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(
                candidate_stdout=scaled_fixture(CANONICAL, factor=factor,
                                                build_commit="cafe12345"),
                anchor_stdout=read_fixture(CANONICAL)),
            host_state=HostStateStub([healthy_state()]))
        return runner.run(make_plan(self.binding, blocks=8))

    def test_the_blocks_are_statistics_paired_blocks(self):
        for block in self._run(1.08).paired_blocks():
            self.assertIsInstance(block, statistics.PairedBlock)
            block.to_tuple()
            schemas.canonical_json({"b": block.to_list()})

    def test_block_effect_reads_the_expected_direction_and_magnitude(self):
        blocks = self._run(1.08).paired_blocks()
        effects = [statistics.block_effect(b, scale=statistics.EFFECT_SCALE_RELATIVE)
                   for b in blocks]
        self.assertTrue(all(e > 0 for e in effects), effects)
        self.assertAlmostEqual(statistics.median(tuple(effects)), 0.08, places=2)

    def test_a_slower_candidate_reads_negative(self):
        """Metric direction control: the sign must not be hardcoded optimistic."""
        blocks = self._run(0.92).paired_blocks()
        effects = [statistics.block_effect(b, scale=statistics.EFFECT_SCALE_RELATIVE)
                   for b in blocks]
        self.assertTrue(all(e < 0 for e in effects), effects)

    def test_an_aa_run_reads_as_no_effect(self):
        blocks = self._run(1.0).paired_blocks()
        effects = [statistics.block_effect(b, scale=statistics.EFFECT_SCALE_RELATIVE)
                   for b in blocks]
        self.assertTrue(all(abs(e) < 1e-9 for e in effects), effects)

    def test_the_blocks_carry_the_unit_stratum_and_segment_the_reducer_partitions_on(self):
        for block in self._run(1.08).paired_blocks():
            self.assertEqual(block.unit_id, "unit-0")
            self.assertEqual(block.stratum, api.STRATUM_SELECTION)
            self.assertEqual(block.segment, statistics.SEGMENT_BASE)
            self.assertIsNone(block.extension_round)
            self.assertIsNotNone(block.measured_at)


# =============================================================================
# Process discipline
# =============================================================================

class TestProcessDisciplineIsStructural(unittest.TestCase):

    def test_the_module_cannot_find_a_process_by_name(self):
        check = M.audit_no_name_pattern_process_paths()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_audit_catches_a_name_pattern_kill(self):
        """The guard's own bite. Without this the PASS above proves nothing."""
        source = ('import subprocess\n'
                  'def cleanup():\n'
                  '    subprocess.run(["pkill", "-f", "llama-server"])\n')
        check = M.audit_no_name_pattern_process_paths(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("pkill" in r for r in check.reasons))

    def test_the_audit_catches_a_shelled_out_pattern_kill(self):
        source = ('import subprocess\n'
                  'subprocess.run("pgrep -f llama | xargs kill", shell=True)\n')
        check = M.audit_no_name_pattern_process_paths(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("shell=True" in r for r in check.reasons))

    def test_the_audit_catches_os_kill_on_an_arbitrary_pid(self):
        source = 'import os\ndef stop(pid):\n    os.kill(pid, 9)\n'
        check = M.audit_no_name_pattern_process_paths(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("os.kill" in r for r in check.reasons))

    def test_the_audit_catches_the_signal_module(self):
        source = 'import signal\n'
        check = M.audit_no_name_pattern_process_paths(source)
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_the_audit_does_not_forbid_its_own_vocabulary(self):
        """A guard that fails on the constant defining the ban gets deleted.

        Naming `pkill` in a docstring, a constant or a reason string must remain
        legal; only handing it to a launcher is not.
        """
        source = ('FORBIDDEN = frozenset({"pkill", "pgrep", "killall"})\n'
                  '"""Never run pkill; see INC-20260731."""\n'
                  'MESSAGE = "do not use pkill or pgrep on this host"\n')
        check = M.audit_no_name_pattern_process_paths(source)
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_audit_permits_spawning_a_constructed_argv(self):
        """Compliant control: the runner's own idiom must pass."""
        source = ('import subprocess\n'
                  'def run(argv, env):\n'
                  '    p = subprocess.Popen(argv, env=env)\n'
                  '    p.terminate()\n'
                  '    p.kill()\n'
                  '    return p.wait()\n')
        check = M.audit_no_name_pattern_process_paths(source)
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_spawner_never_constructs_a_pipe_for_the_tool(self):
        """*"Never pipe llama binaries through another process."*"""
        source = Path(M.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        piped = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "PIPE"
            and isinstance(node.value, ast.Name) and node.value.id == "subprocess"
        ]
        self.assertEqual(piped, [], "stdout/stderr must go to files, never a pipe")

    def test_the_spawner_passes_an_explicit_env_so_nothing_ambient_leaks(self):
        source = Path(M.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        popens = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and M._callee_name(n) == "Popen"]
        self.assertTrue(popens)
        for call in popens:
            self.assertIn("env", [kw.arg for kw in call.keywords],
                          "Popen must receive an explicit env")


class TestProcessDisciplineActuallyTerminates(unittest.TestCase):
    """The one place a real process is spawned. `/bin/sleep`, not inference.

    A termination path that is never exercised is a termination path that does
    not work. This spawns a process THIS TEST launched, times it out, and then
    verifies with `ps -p <pid>` that it is genuinely gone — the project rule is
    "never report success until confirmed", and the confirmation is the point.
    """

    @unittest.skipUnless(Path("/bin/sleep").exists(), "no /bin/sleep")
    def test_a_timeout_terminates_the_captured_pid_and_confirms_it_is_dead(self):
        spawner = M.SubprocessSpawner(term_grace_s=5.0)
        result = spawner.run(["/bin/sleep", "30"], {"PATH": "/usr/bin:/bin"},
                             timeout_s=0.5)
        self.assertTrue(result.timed_out)
        self.assertTrue(result.terminated_by_runner)
        self.assertIsNotNone(result.pid)
        probe = subprocess.run(["ps", "-p", str(result.pid)], capture_output=True,
                               text=True, check=False)
        self.assertNotEqual(probe.returncode, 0,
                            f"pid {result.pid} survived: {probe.stdout}")

    @unittest.skipUnless(Path("/bin/echo").exists(), "no /bin/echo")
    def test_stdout_is_captured_from_a_file_and_returned_whole(self):
        spawner = M.SubprocessSpawner()
        result = spawner.run(["/bin/echo", "hello-autokernel"], {"PATH": "/usr/bin:/bin"},
                             timeout_s=30)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "hello-autokernel")
        self.assertFalse(result.timed_out)
        self.assertFalse(result.terminated_by_runner)

    @unittest.skipUnless(Path("/usr/bin/env").exists(), "no /usr/bin/env")
    def test_the_child_receives_only_the_env_it_was_given(self):
        """Structural proof that `os.environ` does not leak into a measurement."""
        os.environ["AUTOKERNEL_LEAK_CANARY"] = "leaked"
        self.addCleanup(os.environ.pop, "AUTOKERNEL_LEAK_CANARY", None)
        spawner = M.SubprocessSpawner()
        result = spawner.run(["/usr/bin/env"], {"PATH": "/usr/bin:/bin",
                                                "OMP_PROC_BIND": "spread"}, timeout_s=30)
        self.assertIn("OMP_PROC_BIND=spread", result.stdout)
        self.assertNotIn("AUTOKERNEL_LEAK_CANARY", result.stdout)

    def test_a_missing_binary_raises_rather_than_reporting_a_zero(self):
        spawner = M.SubprocessSpawner()
        with self.assertRaises(M.SpawnFailure):
            spawner.run(["/nonexistent/autokernel/binary"], {"PATH": "/usr/bin"},
                        timeout_s=5)


# =============================================================================
# Frozen-tree and scope guards
# =============================================================================

class TestFrozenTreesAndScope(unittest.TestCase):

    def test_a_candidate_bound_to_a_production_tree_is_refused_upstream(self):
        """Denial 2, delegated to `recipes` — this test proves the delegation holds."""
        production = recipes.storage.production_tree_forms()
        self.assertTrue(production, "no production tree forms are declared")
        root = Path(sorted(production)[0])
        binding = recipes.ToolBinding(binary=str(root / "bin" / "llama-bench"),
                                      source_root=str(root),
                                      library_path=str(root / "bin"))
        with self.assertRaises(recipes.RecipeBindingError):
            recipes.construct(RECIPE_ID, binding=binding, params=default_params(),
                              arm=M.ARM_CANDIDATE, verify_inputs=False)

    def test_the_anchor_arm_may_read_the_frozen_tree(self):
        """Compliant control: executing the frozen anchor read-only is not a write."""
        production = recipes.storage.production_tree_forms()
        root = Path(sorted(production)[0])
        binding = recipes.ToolBinding(binary=str(root / "bin" / "llama-bench"),
                                      source_root=str(root),
                                      library_path=str(root / "bin"))
        command = recipes.construct(RECIPE_ID, binding=binding, params=default_params(),
                                    arm=M.ARM_ANCHOR, verify_inputs=False)
        self.assertEqual(command.arm, M.ARM_ANCHOR)

    def test_the_claim_footprint_is_derived_from_the_argv_taskset_mask(self):
        binding = BindingFixture(self)
        command = build_command(binding.candidate)
        self.assertEqual(command.claim_footprint.cpu_list, "0-95")
        self.assertEqual(command.claim_footprint.cpu_count, 96)
        self.assertEqual(command.scope_denominator.cores, 96)

    def test_the_scope_denominator_travels_with_the_samples(self):
        """A full-machine gate on a partial-machine cell is a category error."""
        binding = BindingFixture(self)
        runner = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=read_fixture(CANONICAL),
                                      anchor_stdout=read_fixture(CANONICAL)),
            host_state=HostStateStub([healthy_state()]))
        vector = runner.run(make_plan(binding, blocks=1)).raw_vector()
        gate_scope = {"machine_subset": "full", "numa_nodes": [], "devices": [],
                      "cores": 96}
        self.assertEqual(
            schemas.check_scope_denominator_admits_gate(vector, gate_scope).outcome,
            schemas.PASS)

    def test_a_full_machine_gate_is_refused_on_a_narrower_cell(self):
        """The bite: the denominator must be able to REFUSE a gate."""
        binding = BindingFixture(self)
        runner = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=read_fixture(CANONICAL),
                                      anchor_stdout=read_fixture(CANONICAL)),
            host_state=HostStateStub([healthy_state()]))
        vector = runner.run(make_plan(binding, blocks=1)).raw_vector()
        vector["scope_denominator"]["cores"] = 48
        gate_scope = {"machine_subset": "full", "numa_nodes": [], "devices": [],
                      "cores": 96}
        self.assertEqual(
            schemas.check_scope_denominator_admits_gate(vector, gate_scope).outcome,
            schemas.FAIL)


class TestCpuRegionClaimAdapter(unittest.TestCase):
    """The adapter that makes this runner usable tomorrow without edits.

    The claim objects here hold REAL `flock`s on REAL files, because the defect
    this class exists to keep closed is precisely that `CpuRegionClaim.held` is
    `not self._released` — an in-process boolean that no external event can move.
    A fake whose `held` attribute the test flips proves the adapter re-reads a
    PYTHON ATTRIBUTE, which is what the previous version of this class proved,
    and is not the property under test.
    """

    #: A narrow footprint, so the fixture locks one atomic region rather than the
    #: whole machine. Nothing here dispatches anything; the lock root is a temp
    #: directory, so these flocks exclude nobody outside this test.
    CLAIM_CPUS = "0-11"

    def setUp(self):
        try:
            from . import cpu_region_claim as C
        except ImportError:                                # pragma: no cover
            self.skipTest("execution/cpu_region_claim.py is not present")
        self.C = C
        self.tmp = tempfile.TemporaryDirectory(prefix="autokernel-claim-test-")
        self.addCleanup(self.tmp.cleanup)
        self.lock_root = Path(self.tmp.name)
        try:
            self.plan = C.plan_region_claim(self.CLAIM_CPUS, role="autokernel",
                                            lock_root=str(self.lock_root))
        except C.CpuTopologyUnavailable:                   # pragma: no cover
            self.skipTest("this host exposes no thread-sibling topology")

    def _receipt(self, claim_id: str, *, expires_at=None, lock_paths=None) -> dict:
        """A SELF-CONSISTENT receipt, derived by the sibling's own planner.

        Hand-writing the fields does not work and should not: the sibling
        re-derives `regions`, `lock_paths` and `physical_core_list` from
        `cpu_list` and refuses a receipt that contradicts its own derivation.
        """
        return {
            "claim_id": claim_id, "role": self.plan.role, "roles": list(self.plan.roles),
            "cpu_list": self.plan.cpu_list,
            "physical_core_list": self.C.render_cpu_list(self.plan.physical_cores),
            "regions": list(self.plan.regions),
            "lock_paths": ([str(p) for _r, _g, p in self.plan.lock_steps]
                           if lock_paths is None else [str(p) for p in lock_paths]),
            "lock_root": self.plan.lock_root, "state": "held",
            "holder_pid": os.getpid(), "holder_start_ticks": 1, "holder_boot_id": "boot",
            "host": "test-host", "purpose": "red-team fixture", "campaign_id": "camp",
            "acquired_at": "2026-08-03T22:00:00+00:00", "holder_label": None,
            "expires_at": expires_at, "released_at": None, "reclaimed_from": None,
            "schema": self.C.RECEIPT_SCHEMA,
        }

    def _materialise_locks(self, claim_id: str, *, flocked: bool = True,
                           payload_claim_id: str | None = None) -> None:
        """Create every lock file the plan names, optionally holding real flocks."""
        import fcntl
        for lock_role, region, path in self.plan.lock_steps:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if lock_role == self.C.GLOBAL_MUTEX_ROLE:
                path.write_bytes(b"")          # exclusion-only layer: no payload
            else:
                path.write_text(json.dumps(
                    {"claim_id": payload_claim_id or claim_id, "role": lock_role,
                     "region": region}) + "\n")
            if flocked:
                handle = path.open("r+b")
                self.addCleanup(handle.close)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    class FakeRegionClaim:
        """Shaped like `cpu_region_claim.CpuRegionClaim`, including its receipt."""

        def __init__(self, receipt: dict, *, held=True, covers=True):
            self._receipt = receipt
            self.claim_id = receipt["claim_id"]
            self.held = held
            self._covers = covers
            self.covers_calls = []
            self.receipt_calls = 0

        def covers(self, cpu_list, sibling_map=None):
            self.covers_calls.append(cpu_list)
            return self._covers

        def receipt(self):
            self.receipt_calls += 1
            return dict(self._receipt)

    def _held_claim(self, *, claim_id="akc-test", expires_at=None, **kwargs):
        self._materialise_locks(claim_id)
        return self.FakeRegionClaim(self._receipt(claim_id, expires_at=expires_at),
                                    **kwargs)

    def test_a_held_covering_claim_attests_pass(self):
        adapter = M.CpuRegionClaimAdapter(self._held_claim(), cpu_list=self.CLAIM_CPUS)
        attestation = adapter.attest()
        self.assertTrue(attestation.held, attestation.check.reasons)
        self.assertEqual(attestation.cpu_list, self.CLAIM_CPUS)
        self.assertEqual(adapter.claim_id, "akc-test")

    # -- the bite: the flag and the filesystem can disagree ---------------

    def test_a_claim_whose_flock_is_free_attests_fail_however_held_it_says_it_is(self):
        """`claim.held` is `not self._released`. Nothing external can move it.

        This is the defect. The object reports held, `covers()` agrees, and the
        region locks it names are FREE — leaked, or dropped by a crashed helper.
        An adapter built on `held` alone re-reads a Python attribute 2N times per
        campaign and calls it a mid-run revocation check; every attestation in
        the raw vector then asserts an observation that was never made.
        """
        self._materialise_locks("akc-leaked", flocked=False)
        claim = self.FakeRegionClaim(self._receipt("akc-leaked"))
        self.assertTrue(claim.held)
        self.assertTrue(claim.covers(self.CLAIM_CPUS))
        attestation = M.CpuRegionClaimAdapter(claim, cpu_list=self.CLAIM_CPUS).attest()
        self.assertFalse(attestation.held)
        self.assertTrue(any("FREE" in r for r in attestation.check.reasons),
                        attestation.check.reasons)

    def test_a_lock_now_recorded_to_another_claim_attests_fail(self):
        self._materialise_locks("akc-mine", payload_claim_id="akc-somebody-else")
        claim = self.FakeRegionClaim(self._receipt("akc-mine"))
        attestation = M.CpuRegionClaimAdapter(claim, cpu_list=self.CLAIM_CPUS).attest()
        self.assertFalse(attestation.held)

    def test_a_claim_naming_no_locks_attests_fail(self):
        claim = self.FakeRegionClaim(self._receipt("akc-empty", lock_paths=()))
        attestation = M.CpuRegionClaimAdapter(claim, cpu_list=self.CLAIM_CPUS).attest()
        self.assertFalse(attestation.held)

    def test_an_expired_claim_attests_fail(self):
        claim = self._held_claim(claim_id="akc-expired",
                                 expires_at="2020-01-01T00:00:00+00:00")
        attestation = M.CpuRegionClaimAdapter(claim, cpu_list=self.CLAIM_CPUS).attest()
        self.assertFalse(attestation.held)
        self.assertTrue(any("expired" in r for r in attestation.check.reasons))

    def test_a_claim_declaring_no_expiry_is_still_held(self):
        """Compliant control: COULD_NOT_CHECK from the expiry checker is not a failure.

        No `expires_at` is a property of the claim's DECLARATION, not evidence
        that the claim has gone. Failing on it would refuse every claim taken
        without `max_hold_s` and the check would be switched off within a week.
        """
        attestation = M.CpuRegionClaimAdapter(self._held_claim(),
                                              cpu_list=self.CLAIM_CPUS).attest()
        self.assertTrue(attestation.held, attestation.check.reasons)

    def test_the_lock_is_re_read_on_every_attestation_not_once(self):
        claim = self._held_claim()
        adapter = M.CpuRegionClaimAdapter(claim, cpu_list=self.CLAIM_CPUS)
        adapter.attest()
        adapter.attest()
        adapter.attest()
        self.assertEqual(claim.receipt_calls, 3)

    def test_a_claim_that_cannot_produce_a_receipt_cannot_be_adapted(self):
        class NoReceipt:
            claim_id = "akc-x"
            held = True

            def covers(self, cpu_list, sibling_map=None):
                return True

        with self.assertRaises(TypeError):
            M.CpuRegionClaimAdapter(NoReceipt(), cpu_list=self.CLAIM_CPUS)

    def test_the_real_module_still_exposes_the_re_read_this_depends_on(self):
        """The adapter's whole correctness is these two functions. Fail HERE if they go."""
        try:
            from . import cpu_region_claim as C
        except ImportError:
            self.skipTest("execution/cpu_region_claim.py is not present")
        for name in ("check_region_claim_held", "check_claim_expiry"):
            self.assertTrue(callable(getattr(C, name, None)),
                            f"cpu_region_claim lost {name!r}; CpuRegionClaimAdapter "
                            f"re-reads the lock through it and would silently fall back "
                            f"to the in-process `held` flag")

    def test_a_released_claim_attests_fail(self):
        adapter = M.CpuRegionClaimAdapter(self._held_claim(held=False),
                                          cpu_list=self.CLAIM_CPUS)
        self.assertFalse(adapter.attest().held)

    def test_a_non_covering_claim_attests_fail(self):
        adapter = M.CpuRegionClaimAdapter(self._held_claim(covers=False),
                                          cpu_list=self.CLAIM_CPUS)
        attestation = adapter.attest()
        self.assertFalse(attestation.held)
        self.assertTrue(any("exact footprint" in r.lower()
                            for r in attestation.check.reasons))

    def test_the_adapter_re_reads_the_claim_on_every_attestation(self):
        """A cached PASS would defeat the whole mid-run revocation check."""
        claim = self._held_claim()
        adapter = M.CpuRegionClaimAdapter(claim, cpu_list=self.CLAIM_CPUS)
        adapter.attest()
        adapter.attest()
        claim.held = False
        self.assertFalse(adapter.attest().held)
        self.assertEqual(len(claim.covers_calls), 2)

    def test_an_object_of_the_wrong_shape_is_refused_loudly(self):
        with self.assertRaises(TypeError):
            M.CpuRegionClaimAdapter(object(), cpu_list="0-95")

    def test_the_adapter_satisfies_the_runner_without_any_edit_to_the_runner(self):
        binding = BindingFixture(self)
        runner = M.MicrobenchRunner(
            claim=M.CpuRegionClaimAdapter(self._held_claim(), cpu_list="0-95"),
            policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=read_fixture(CANONICAL),
                                      anchor_stdout=read_fixture(CANONICAL)),
            host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(binding, blocks=2))
        self.assertTrue(run.complete, run.refusals)

    def test_the_real_cpu_region_claim_module_still_has_the_shape_this_adapts(self):
        """If the sibling module renames `held`/`covers`, fail HERE, not tomorrow."""
        try:
            from . import cpu_region_claim as C
        except ImportError:
            self.skipTest("execution/cpu_region_claim.py is not present")
        self.assertTrue(hasattr(C, "CpuRegionClaim"))
        for attribute in ("claim_id", "held", "covers"):
            self.assertTrue(hasattr(C.CpuRegionClaim, attribute),
                            f"CpuRegionClaim lost {attribute!r}; "
                            f"CpuRegionClaimAdapter must be updated")


class TestPlanValidation(unittest.TestCase):

    def test_the_anchor_must_be_a_named_identity_not_a_path(self):
        binding = BindingFixture(self)
        with self.assertRaises(TypeError):
            M.MicrobenchPlan(
                recipe_id=RECIPE_ID, candidate_id="c", campaign_seed="s",
                candidate_binding=binding.candidate, anchor_binding=binding.anchor,
                anchor="/mnt/raid0/llm/llama.cpp", params=default_params(),
                base_blocks=1, pairs_per_block=1, unit_ids=("u",))

    def test_a_campaign_seed_is_required(self):
        binding = BindingFixture(self)
        with self.assertRaises(ValueError):
            M.MicrobenchPlan(
                recipe_id=RECIPE_ID, candidate_id="c", campaign_seed="  ",
                candidate_binding=binding.candidate, anchor_binding=binding.anchor,
                anchor=api.AnchorIdentity(source_commit=ANCHOR_COMMIT,
                                          binary_sha256="a" * 64,
                                          linkage_sha256="b" * 64),
                params=default_params(), base_blocks=1, pairs_per_block=1,
                unit_ids=("u",))

    def test_at_least_one_measurement_material_unit_is_required(self):
        binding = BindingFixture(self)
        with self.assertRaises(ValueError):
            make_plan(binding, blocks=1)._replace_units() if False else \
                M.MicrobenchPlan(
                    recipe_id=RECIPE_ID, candidate_id="c", campaign_seed="s",
                    candidate_binding=binding.candidate, anchor_binding=binding.anchor,
                    anchor=api.AnchorIdentity(source_commit=ANCHOR_COMMIT,
                                              binary_sha256="a" * 64,
                                              linkage_sha256="b" * 64),
                    params=default_params(), base_blocks=1, pairs_per_block=1,
                    unit_ids=())



# =============================================================================
# RED TEAM — regressions for defects found by adversarial review of this module.
#
# Each class below names a way this executor could emit evidence that overstates
# what it did, or could lose its own durable record. Each has a compliant-path
# control beside it: a guard that also refuses the honest case is a guard that
# gets deleted.
# =============================================================================

class TestTheAnchorTripleIsVerifiedNotAsserted(unittest.TestCase):
    """`api.AnchorIdentity` is a TRIPLE. Only one of the three was ever checked."""

    def setUp(self):
        self.binding = BindingFixture(self)
        self.out = read_fixture(CANONICAL)

    def _run(self, plan):
        return M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=self.out, anchor_stdout=self.out),
            host_state=HostStateStub([healthy_state()])).run(plan)

    def test_an_anchor_binding_that_is_not_the_named_anchor_binary_is_refused(self):
        """THE BITE: `anchor_binding` is exempt from the production-tree refusal.

        `recipes._assert_arm_allows_binding` returns immediately for the anchor
        arm — reading the frozen binary is not a write — so NOTHING constrains
        where an anchor binding points. The only thing that can catch a second
        candidate build masquerading as the anchor is the digest the plan names,
        and the runner already computes it on the way past.
        """
        plan = make_plan(self.binding, blocks=1,
                         anchor=anchor_identity(self.binding, binary_sha256="a" * 64))
        run = self._run(plan)
        self.assertFalse(run.complete)
        self.assertTrue(any("anchor_identity.binary_sha256" in r for r in run.refusals),
                        run.refusals)
        with self.assertRaises(M.RunRefused):
            run.paired_blocks()

    def test_the_declared_anchor_binary_digest_passes_when_it_is_the_binary(self):
        run = self._run(make_plan(self.binding, blocks=1))
        self.assertTrue(run.complete, run.refusals)
        checks = dict(run.checks)
        self.assertEqual(checks["anchor_identity.binary_sha256"].outcome, schemas.PASS)

    def test_the_unverified_linkage_conjunct_is_recorded_rather_than_omitted(self):
        """An unverified conjunct that appears nowhere reads exactly like a pass."""
        vector = self._run(make_plan(self.binding, blocks=1)).raw_vector()
        named = dict((n, c) for n, c in run_checks(vector))
        self.assertEqual(named["anchor_identity.linkage_sha256"]["outcome"],
                         schemas.COULD_NOT_CHECK)
        self.assertTrue(any("verify_ggml_linkage" in r
                            for r in named["anchor_identity.linkage_sha256"]["reasons"]))

    def test_the_anchor_triple_travels_with_the_evidence(self):
        vector = self._run(make_plan(self.binding, blocks=1)).raw_vector()
        self.assertEqual(vector["anchor_identity"]["binary_sha256"],
                         self._run(make_plan(self.binding, blocks=1))
                         .anchor_receipt.binary_sha256)


def run_checks(vector: dict):
    return [(name, payload) for name, payload in vector["checks"]]


class TestAVacuousBuildCommitMatchIsNotAMatch(unittest.TestCase):
    """`actual.startswith("")` is TRUE for every anchor in existence."""

    def _row(self, build_commit: str) -> M.BenchRow:
        rows = json.loads(read_fixture(CANONICAL))
        rows[0]["build_commit"] = build_commit
        return M.parse_llama_bench_json(json.dumps(rows))[0]

    def _expect(self, expected: str) -> M.LlamaBenchExpectation:
        binding = BindingFixture(self)
        return M.LlamaBenchExpectation.from_command(build_command(binding.anchor,
                                                                  arm=M.ARM_ANCHOR),
                                                    expected_build_commit=expected)

    def test_an_empty_build_commit_does_not_satisfy_every_anchor(self):
        check = self._expect("dead" + "beef" * 9).check_row(self._row(""))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("the anchor is not the anchor" in r for r in check.reasons))

    def test_a_one_character_build_commit_does_not_satisfy_one_anchor_in_sixteen(self):
        check = self._expect("6" + "a" * 39).check_row(self._row("6"))
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_a_non_hex_build_commit_is_not_a_commit(self):
        check = self._expect("dead" + "beef" * 9).check_row(self._row("unknown"))
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_the_real_abbreviated_prefix_still_matches(self):
        """Compliant control: prefix matching is REQUIRED, it just may not be vacuous."""
        check = self._expect(ANCHOR_COMMIT).check_row(self._row(FIXTURE_BUILD_COMMIT))
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_helper_states_both_directions(self):
        self.assertTrue(M.commit_prefix_match("91745611f" + "0" * 31, "91745611f")[0])
        self.assertTrue(M.commit_prefix_match("91745611f", "91745611f" + "0" * 31)[0])
        self.assertFalse(M.commit_prefix_match("91745611f" + "0" * 31, "")[0])
        self.assertFalse(M.commit_prefix_match("", "91745611f")[0])


class TestAMalformedRowIsARefusalNotACrash(unittest.TestCase):
    """The producer of these rows is the process being measured. It must not be
    able to choose which exception its consumer raises."""

    def _row_with(self, key, value) -> str:
        rows = json.loads(read_fixture(CANONICAL))
        rows[0][key] = value
        return json.dumps(rows)

    def test_a_non_numeric_thread_count_is_a_bench_output_error(self):
        with self.assertRaises(M.BenchOutputError):
            M.parse_llama_bench_json(self._row_with("n_threads", "ninety-six"))

    def test_a_non_numeric_average_is_a_bench_output_error(self):
        with self.assertRaises(M.BenchOutputError):
            M.parse_llama_bench_json(self._row_with("avg_ts", "fast"))

    def test_a_null_prompt_length_is_a_bench_output_error(self):
        with self.assertRaises(M.BenchOutputError):
            M.parse_llama_bench_json(self._row_with("n_prompt", None))

    def test_a_non_numeric_nanosecond_sample_is_a_bench_output_error(self):
        with self.assertRaises(M.BenchOutputError):
            M.parse_llama_bench_json(self._row_with("samples_ns", ["x"]))

    def test_a_non_finite_average_is_a_bench_output_error(self):
        with self.assertRaises(M.BenchOutputError):
            M.parse_llama_bench_json(self._row_with("avg_ts", "inf"))

    def test_the_real_row_still_parses(self):
        """Compliant control: every field of the recorded run reads normally."""
        row = M.parse_llama_bench_json(read_fixture(CANONICAL))[0]
        self.assertEqual(row.n_threads, 96)
        self.assertEqual(len(row.samples_ts), FIXTURE_REPS)

    def test_the_runner_refuses_rather_than_raising_on_a_malformed_row(self):
        """THE BITE: a `ValueError` out of `run()` takes the durable record with it."""
        binding = BindingFixture(self)
        runner = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(
                candidate_stdout=self._row_with("n_threads", "ninety-six"),
                anchor_stdout=read_fixture(CANONICAL)),
            host_state=HostStateStub([healthy_state()]))
        run = runner.run(make_plan(binding, blocks=2))
        self.assertFalse(run.complete)
        self.assertTrue(run.refusals)
        self.assertTrue(run.raw_vector()["blocks"])


class TestAFailureIsDurable(unittest.TestCase):
    """*"A failure that is not durable is indistinguishable from a run that never
    happened."* An exception out of `run()` is exactly that."""

    class ExplodingSpawner:
        spawner_id = "exploding/v1"

        def __init__(self, exc, *, after: int = 0):
            self.exc = exc
            self.after = after
            self.calls = 0

        def run(self, argv, env, *, timeout_s, cwd=None):
            self.calls += 1
            if self.calls > self.after:
                raise self.exc
            return M.SpawnResult(argv=tuple(argv), returncode=0,
                                 stdout=read_fixture(CANONICAL), stderr_tail="",
                                 pid=None, duration_s=0.0)

    def _runner(self, spawner):
        return M.MicrobenchRunner(claim=StubClaim(), policy=HEALTHY_POLICY,
                                  spawner=spawner,
                                  host_state=HostStateStub([healthy_state()]))

    def test_a_spawn_that_cannot_start_refuses_instead_of_raising(self):
        binding = BindingFixture(self)
        run = self._runner(self.ExplodingSpawner(
            M.SpawnFailure("could not start 'llama-bench': ENOENT"))).run(
                make_plan(binding, blocks=2))
        self.assertFalse(run.complete)
        self.assertTrue(any("could not start" in r for r in run.refusals), run.refusals)
        self.assertIn("blocks", run.raw_vector())

    def test_an_os_error_mid_campaign_keeps_the_blocks_already_measured(self):
        binding = BindingFixture(self)
        spawner = self.ExplodingSpawner(OSError("input/output error"), after=2)
        run = self._runner(spawner).run(make_plan(binding, blocks=3))
        self.assertFalse(run.complete)
        self.assertTrue(run.raw_vector()["blocks"],
                        "the first completed block must survive the failure")
        with self.assertRaises(M.RunRefused):
            run.paired_blocks()

    def test_a_spawner_that_breaks_its_contract_still_raises(self):
        """Compliant control in the other direction: a caller-side DEFECT is not
        a fact about the host and must not be laundered into a refusal."""
        class WrongType:
            spawner_id = "wrong"

            def run(self, argv, env, *, timeout_s, cwd=None):
                return "some stdout"

        binding = BindingFixture(self)
        with self.assertRaises(TypeError):
            self._runner(WrongType()).run(make_plan(binding, blocks=1))

    def test_the_healthy_run_is_unaffected(self):
        binding = BindingFixture(self)
        out = read_fixture(CANONICAL)
        run = self._runner(arm_aware_spawner(candidate_stdout=out,
                                             anchor_stdout=out)).run(
            make_plan(binding, blocks=2))
        self.assertTrue(run.complete, run.refusals)


class TestTheBinaryIsRedigestedBeforeEveryInvocation(unittest.TestCase):
    """*"A path is not an identity on a host where the experimental tree is
    rebuilt between blocks."* This loop's whole purpose is rebuilding."""

    class RebuildingSpawner:
        """Rewrites the candidate binary after the first invocation."""

        spawner_id = "rebuilding/v1"

        def __init__(self, binary: Path, stdout: str, *, after: int = 1):
            self.binary = binary
            self.stdout = stdout
            self.after = after
            self.calls = 0

        def run(self, argv, env, *, timeout_s, cwd=None):
            self.calls += 1
            if self.calls == self.after:
                self.binary.write_bytes(b"candidate-build-bytes-REBUILT")
            return M.SpawnResult(argv=tuple(argv), returncode=0, stdout=self.stdout,
                                 stderr_tail="", pid=None, duration_s=0.0)

    def test_a_rebuild_mid_campaign_is_caught_at_the_next_invocation(self):
        binding = BindingFixture(self)
        spawner = self.RebuildingSpawner(Path(binding.candidate.binary),
                                         read_fixture(CANONICAL))
        run = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY, spawner=spawner,
            host_state=HostStateStub([healthy_state()])).run(
                make_plan(binding, blocks=2))
        self.assertFalse(run.complete)
        self.assertTrue(any("changed" in r for r in run.refusals), run.refusals)

    def test_a_binary_that_vanishes_mid_campaign_refuses(self):
        binding = BindingFixture(self)

        class Deleting(TestTheBinaryIsRedigestedBeforeEveryInvocation.RebuildingSpawner):
            def run(self, argv, env, *, timeout_s, cwd=None):
                self.calls += 1
                if self.calls == self.after:
                    self.binary.unlink()
                return M.SpawnResult(argv=tuple(argv), returncode=0, stdout=self.stdout,
                                     stderr_tail="", pid=None, duration_s=0.0)

        run = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=Deleting(Path(binding.candidate.binary), read_fixture(CANONICAL)),
            host_state=HostStateStub([healthy_state()])).run(
                make_plan(binding, blocks=2))
        self.assertFalse(run.complete)
        self.assertTrue(any("could not be digested" in r for r in run.refusals),
                        run.refusals)

    def test_an_unchanged_binary_runs_every_block(self):
        """Compliant control: the digest check must not refuse a stable tree."""
        binding = BindingFixture(self)
        out = read_fixture(CANONICAL)
        run = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=out, anchor_stdout=out),
            host_state=HostStateStub([healthy_state()])).run(
                make_plan(binding, blocks=3))
        self.assertTrue(run.complete, run.refusals)


class TestPreRegistrationIsEvidencedNotAsserted(unittest.TestCase):
    """A hardcoded `campaign_seed_committed: True` is a field nothing can contradict."""

    def _vector(self, blocks=3):
        binding = BindingFixture(self)
        out = read_fixture(CANONICAL)
        runner = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=out, anchor_stdout=out),
            host_state=HostStateStub([healthy_state()]))
        self.plan = make_plan(binding, blocks=blocks)
        self.run_ = runner.run(self.plan)
        return self.run_.raw_vector()

    def test_the_unfalsifiable_assertion_is_gone(self):
        self.assertNotIn("campaign_seed_committed", self._vector())

    def test_the_seed_digest_and_the_derived_schedule_ship_with_the_samples(self):
        vector = self._vector()
        self.assertEqual(
            vector["campaign_seed_sha256"],
            hashlib.sha256(self.plan.campaign_seed.encode("utf-8")).hexdigest())
        for key, value in self.plan.schedule().to_dict().items():
            self.assertEqual(vector["order_schedule"][key], value)

    def test_a_reader_of_the_raw_vector_can_check_the_order_control_without_the_seed(self):
        """The point of shipping the schedule: the claim becomes checkable.

        `OrderSchedule.to_dict()` withholds the campaign seed on purpose, so the
        required orders are spelled out and the reader binds them to the
        committed campaign record through `campaign_seed_sha256`.
        """
        vector = self._vector()
        required = vector["order_schedule"]["orders"]
        self.assertEqual(len(required), self.plan.base_blocks)
        observed = [block["plan"]["order"] for block in vector["blocks"]]
        self.assertEqual(observed, required)
        self.assertEqual([b.order for b in self.run_.paired_blocks()], required)

    def test_the_emitted_order_control_is_the_reducers_own_verdict(self):
        vector = self._vector()
        self.assertEqual(vector["order_control"]["outcome"], schemas.PASS)
        self.assertNotIn("order_schedule_control", [n for n, _ in vector["checks"]],
                         "a frozen COPY of the control is a second thing to stub; "
                         "`order_control` must be the only computation site")

    def _seed_drawing_a_different_schedule(self) -> str:
        """A seed whose schedule really differs, established WITHOUT the property
        under test. Picked here rather than by asking `order_control`, so a
        stubbed property cannot make this test skip itself into silence."""
        mine = statistics.OrderSchedule.derive(
            campaign_seed=self.plan.campaign_seed, candidate_id=self.plan.candidate_id,
            base_blocks=self.plan.base_blocks).orders(self.plan.base_blocks)
        for n in range(200):
            seed = f"a-different-committed-seed-{n}"
            other = statistics.OrderSchedule.derive(
                campaign_seed=seed, candidate_id=self.plan.candidate_id,
                base_blocks=self.plan.base_blocks).orders(self.plan.base_blocks)
            if other != mine:
                return seed
        raise AssertionError("no seed in 200 drew a different schedule; the schedule "
                             "is not a function of the seed")

    def test_a_run_cannot_be_relabelled_with_a_different_campaign_seed(self):
        """THE BITE, and why `order_control` is a property and not a frozen value.

        Inside the runner the control is a TAUTOLOGY: it runs the schedule it
        derived, so a verdict computed once during the run can only ever say
        PASS, and a hardcoded PASS is indistinguishable from a real one. The
        property that is not a tautology is that these blocks were produced under
        THIS plan — so restapling a plan with a different committed seed onto a
        completed run must stop the blocks coming out.
        """
        self._vector()
        relabelled = replace(
            self.run_,
            plan=replace(self.plan, campaign_seed=self._seed_drawing_a_different_schedule()))
        self.assertEqual(relabelled.order_control.outcome, schemas.FAIL)
        self.assertFalse(relabelled.complete)
        with self.assertRaises(M.RunRefused):
            relabelled.paired_blocks()
        self.assertEqual(relabelled.raw_vector()["order_control"]["outcome"],
                         schemas.FAIL)

    def test_relabelling_with_the_same_seed_is_still_admissible(self):
        """Compliant control: the check must key on the seed, not on identity."""
        self._vector()
        same = replace(self.run_,
                       plan=replace(self.plan, campaign_seed=self.plan.campaign_seed))
        self.assertTrue(same.complete, same.refusals)

    def test_a_block_set_that_contradicts_the_schedule_is_refused(self):
        """THE BITE: the control must be able to FAIL, not merely be present."""
        binding = BindingFixture(self)
        out = read_fixture(CANONICAL)
        runner = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=out, anchor_stdout=out),
            host_state=HostStateStub([healthy_state()]))
        plan = make_plan(binding, blocks=3)
        wrong = statistics.OrderSchedule.derive(
            campaign_seed="a-different-campaign-seed", candidate_id=plan.candidate_id,
            base_blocks=plan.base_blocks, attempt=plan.attempt)
        real = plan.schedule()
        if all(wrong.order_for(i) == real.order_for(i) for i in range(plan.base_blocks)):
            self.skipTest("the two seeds happen to agree on every block")
        run = runner.run(plan)
        blocks = run.paired_blocks()
        self.assertEqual(wrong.check_observed(blocks).outcome, schemas.FAIL)


class TestTheDelegatedObligationsAreVisible(unittest.TestCase):
    """`recipes` emits COULD_NOT_CHECK findings saying *the runner MUST* do X.
    `check_recipe_discipline` refuses only on FAIL, so they rendered as nothing."""

    def _vector(self):
        binding = BindingFixture(self)
        out = read_fixture(CANONICAL)
        runner = M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=out, anchor_stdout=out),
            host_state=HostStateStub([healthy_state()]))
        return runner.run(make_plan(binding, blocks=1)).raw_vector()

    def test_the_constructor_really_does_delegate_these(self):
        """Precondition for the test below: these findings exist upstream."""
        binding = BindingFixture(self)
        ids = {f.finding_id for f in build_command(binding.candidate).discipline
               if f.check.outcome != schemas.PASS}
        self.assertIn("binary_linkage_resolution", ids)

    def test_every_undischarged_delegation_appears_in_the_emitted_evidence(self):
        names = [n for n, _ in self._vector()["checks"]]
        self.assertIn("delegated.candidate.binary_linkage_resolution", names)
        self.assertIn("delegated.anchor.binary_linkage_resolution", names)

    def test_they_are_recorded_as_could_not_check_and_not_as_passes(self):
        named = dict((n, p) for n, p in self._vector()["checks"])
        self.assertEqual(
            named["delegated.candidate.binary_linkage_resolution"]["outcome"],
            schemas.COULD_NOT_CHECK)

    def test_a_discipline_finding_that_passes_is_not_reported_as_delegated(self):
        """Compliant control: this must not become a dump of every finding."""
        names = [n for n, _ in self._vector()["checks"]]
        binding = BindingFixture(self)
        passing = [f.finding_id for f in build_command(binding.candidate).discipline
                   if f.check.outcome == schemas.PASS]
        for finding_id in passing:
            self.assertNotIn(f"delegated.candidate.{finding_id}", names)


class TestTheProcessAuditCannotBeSpelledAround(unittest.TestCase):
    """A guard that any spelling of its own target walks past is not a guard."""

    def test_an_f_string_does_not_hide_a_name_pattern_kill(self):
        """THE BITE: `f"pkill"` parses as JoinedStr, not Constant."""
        source = ('import subprocess\n'
                  'def go(name):\n'
                  '    subprocess.Popen([f"pkill", "-f", name])\n')
        self.assertEqual(M.audit_no_name_pattern_process_paths(source).outcome,
                         schemas.FAIL)

    def test_a_concatenated_argv_does_not_hide_one(self):
        source = ('import subprocess\n'
                  'def go(name):\n'
                  '    subprocess.Popen(["pgrep"] + ["-f", name])\n')
        self.assertEqual(M.audit_no_name_pattern_process_paths(source).outcome,
                         schemas.FAIL)

    def test_from_os_import_kill_does_not_hide_an_arbitrary_pid_signal(self):
        source = ('from os import kill\n'
                  'def go(pid):\n'
                  '    kill(pid, 9)\n')
        check = M.audit_no_name_pattern_process_paths(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("bare name" in r or "unbinds" in r for r in check.reasons))

    def test_the_import_itself_is_refused_even_when_the_call_is_indirect(self):
        """The import check must bite on its own.

        `from os import killpg` followed by `kill(pid)` is caught twice, so the
        bare-call check alone makes the import check look redundant. Aliasing the
        function into a variable and calling it through that variable defeats
        every call-site check there is; the only thing left is the import.
        """
        source = ('from os import killpg\n'
                  'SIGNALLER = killpg\n'
                  'def go(pgid):\n'
                  '    return SIGNALLER(pgid, 15)\n')
        check = M.audit_no_name_pattern_process_paths(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("unbinds" in r for r in check.reasons), check.reasons)

    def test_the_permitted_termination_path_is_still_permitted(self):
        """Compliant control: `proc.kill()` on a handle we created is the ONE
        allowed spelling, and forbidding it would delete the only safe path."""
        source = ('import subprocess\n'
                  'def go(argv):\n'
                  '    proc = subprocess.Popen(argv)\n'
                  '    proc.terminate()\n'
                  '    proc.kill()\n'
                  '    return proc.wait()\n')
        self.assertEqual(M.audit_no_name_pattern_process_paths(source).outcome,
                         schemas.PASS)

    def test_importing_something_harmless_from_os_is_still_permitted(self):
        source = 'from os import getpid\ndef go():\n    return getpid()\n'
        self.assertEqual(M.audit_no_name_pattern_process_paths(source).outcome,
                         schemas.PASS)

    def test_this_module_still_passes_its_own_audit(self):
        self.assertEqual(M.audit_no_name_pattern_process_paths().outcome, schemas.PASS)



class TestNothingCreatesAFileInAFrozenTree(unittest.TestCase):
    """Hard boundary 1, at the one place this module CREATES anything."""

    PRODUCTION = sorted(recipes.storage.production_tree_forms())[0]

    def test_a_tmpdir_pointing_into_a_frozen_tree_is_refused_before_anything_is_made(self):
        """THE BITE: `tempfile` honours TMPDIR, and the spawner made a directory.

        An `export TMPDIR=<frozen tree>/tmp` in the launching shell is all it
        took to have this module write into the v8 working copy and break the
        `git status --porcelain` byte-identity the boundary is stated in.
        """
        target = Path(self.PRODUCTION) / "autokernel-scratch-must-never-exist"
        with self.assertRaises(M.ProductionTreeWrite):
            M.SubprocessSpawner(workdir_root=str(target))
        self.assertFalse(target.exists(),
                         "the guard must refuse BEFORE creating anything")

    def test_a_symlink_into_a_frozen_tree_is_refused(self):
        """A prefix test on the literal string walks straight past a symlink."""
        tmp = tempfile.TemporaryDirectory(prefix="autokernel-symlink-test-")
        self.addCleanup(tmp.cleanup)
        link = Path(tmp.name) / "innocent-looking-scratch"
        link.symlink_to(self.PRODUCTION)
        with self.assertRaises(M.ProductionTreeWrite):
            M.SubprocessSpawner(workdir_root=str(link))

    def test_a_root_that_contains_a_frozen_tree_is_refused_too(self):
        """Both directions: everything downstream creates beneath the root."""
        with self.assertRaises(M.ProductionTreeWrite):
            M.SubprocessSpawner(workdir_root=str(Path(self.PRODUCTION).parent))

    def test_an_ordinary_scratch_root_is_accepted_and_used(self):
        """Compliant control: the guard must not refuse a normal temp directory."""
        tmp = tempfile.TemporaryDirectory(prefix="autokernel-scratch-test-")
        self.addCleanup(tmp.cleanup)
        spawner = M.SubprocessSpawner(workdir_root=tmp.name)
        result = spawner.run(["/bin/echo", "hello"], {"PATH": "/usr/bin:/bin"},
                             timeout_s=20)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "hello")
        self.assertEqual(list(Path(tmp.name).iterdir()), [],
                         "the workdir must be cleaned up under the root it was given")

    def test_the_default_root_is_checked_and_not_merely_trusted(self):
        """The ambient TMPDIR is exactly the value that would carry the redirect."""
        with self.assertRaises(M.ProductionTreeWrite):
            M.assert_scratch_root_is_not_production(str(Path(self.PRODUCTION) / "tmp"))
        self.assertIsNone(M.assert_scratch_root_is_not_production(None))


class TestTheRecordSaysWhatWasNotVerified(unittest.TestCase):
    """`verify_inputs=False` is a choice; making it invisible is not."""

    def _vector(self):
        binding = BindingFixture(self)
        out = read_fixture(CANONICAL)
        return M.MicrobenchRunner(
            claim=StubClaim(), policy=HEALTHY_POLICY,
            spawner=arm_aware_spawner(candidate_stdout=out, anchor_stdout=out),
            host_state=HostStateStub([healthy_state()])).run(
                make_plan(binding, blocks=1)).raw_vector()

    def test_unverified_inputs_are_recorded_for_both_arms(self):
        named = dict((n, p) for n, p in self._vector()["checks"])
        for arm in (M.ARM_CANDIDATE, M.ARM_ANCHOR):
            self.assertEqual(named[f"inputs_verified.{arm}"]["outcome"],
                             schemas.COULD_NOT_CHECK)

    def test_the_reason_names_the_disabled_verification(self):
        named = dict((n, p) for n, p in self._vector()["checks"])
        self.assertTrue(any("verify_inputs=False" in r for r in
                            named["inputs_verified.candidate"]["reasons"]))

    def test_a_command_whose_inputs_really_were_verified_is_not_flagged(self):
        """Compliant control: this must key on the verification, not be a constant."""
        binding = BindingFixture(self)
        Path(binding.candidate.source_root, ".git").write_text("gitdir: elsewhere\n")
        verified = recipes.construct(RECIPE_ID, binding=binding.candidate,
                                     params=default_params(model=str(binding.model)),
                                     arm=M.ARM_CANDIDATE, verify_inputs=True)
        self.assertTrue(verified.inputs_verified)



class TestADisabledGuardIsNotAPassedGuard(unittest.TestCase):
    """Attack F: does any default or switch yield a SUCCESS-SHAPED result?"""

    def test_disabling_the_frequency_guard_is_could_not_check_not_pass(self):
        policy = M.HostStatePolicy(nominal_khz=3500000, require_frequency=False)
        self.assertEqual(policy.check_frequency(healthy_state()).outcome,
                         schemas.COULD_NOT_CHECK)

    def test_disabling_the_contention_guard_is_could_not_check_not_pass(self):
        policy = M.HostStatePolicy(nominal_khz=3500000, require_load=False)
        self.assertEqual(policy.check_load(healthy_state(), cpu_count=96).outcome,
                         schemas.COULD_NOT_CHECK)

    def test_a_run_under_a_disabled_guard_emits_no_number(self):
        """THE BITE: PASS here made the throttle guard optional at the call site."""
        binding = BindingFixture(self)
        out = read_fixture(CANONICAL)
        run = M.MicrobenchRunner(
            claim=StubClaim(),
            policy=M.HostStatePolicy(nominal_khz=3500000, require_frequency=False),
            spawner=arm_aware_spawner(candidate_stdout=out, anchor_stdout=out),
            host_state=HostStateStub([healthy_state()])).run(
                make_plan(binding, blocks=1))
        self.assertFalse(run.complete)
        self.assertTrue(any("disabled by the caller" in r for r in run.refusals),
                        run.refusals)

    def test_the_enabled_guards_still_pass_a_healthy_host(self):
        """Compliant control: the guards must not have become unconditional."""
        self.assertEqual(HEALTHY_POLICY.check_frequency(healthy_state()).outcome,
                         schemas.PASS)
        self.assertEqual(
            HEALTHY_POLICY.check_load(healthy_state(), cpu_count=96).outcome,
            schemas.PASS)


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
