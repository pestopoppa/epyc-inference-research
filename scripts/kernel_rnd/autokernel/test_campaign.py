#!/usr/bin/env python3
"""test_campaign.py — the driver's guarantees, as tests that BITE.

Every test below either (a) fails if a specific defect is reintroduced, with a
compliant-path control beside it so the guard cannot pass by forbidding its own
idiom, or (b) pins a property the loop's correctness rests on.

NOTHING HERE TOUCHES THE HOST. No process is spawned, no claim is acquired, no
worktree is created, no file outside a `tempfile` tree is written, and no
benchmark is run. The whole suite is `DryRunOps`, a recording spy, and
arithmetic over the recorded A/A numbers.

The five things it proves, in the order they matter:

  1. The dry-run composition walks every step end to end, on a busy host, and
     emits NO speed number.
  2. Every failure path — at every stage — releases the claim and tears down
     the worktree, in that order, and still proves production untouched.
  3. A failed T0 computes no speed number AT ALL: `run_paired_blocks` is never
     called, and `decide()` refuses to be called.
  4. The accept rule accepts a real win and rejects a null, where the null is
     built from the MEASURED A/A drift rather than from an imagined one.
  5. The import boundary is real, and it is enforced against this file's AST
     rather than described in a comment.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import (campaign, journal as journal_module, schemas,
               source_prerequisite_package)
from .execution import control_runner, physical_bounds
from .resource import claim_witness
from .test_schemas import _proposal as _proposal_fixture

MODEL = "/mnt/raid0/llm/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"


def spec(**overrides) -> campaign.CampaignSpec:
    kwargs = dict(campaign_id="ak-test", candidate_id="akc-test",
                  candidate_ref="candidate.patch", model=MODEL)
    kwargs.update(overrides)
    return campaign.CampaignSpec(**kwargs)


def proposal_manifest(campaign_id: str = "ak-test") -> dict:
    proposal = _proposal_fixture()
    proposal["campaign_id"] = campaign_id
    proposal["proposal_id"] = "akp-test-0001"
    proposal["provider_reference"]["target_backend"] = campaign.BACKEND_CPU
    proposal["provider_reference"]["source_commit"] = campaign.MEASUREMENT_COMMIT
    return proposal


def iqk_parameter_proposal(campaign_id: str = "ak-test") -> dict:
    proposal = proposal_manifest(campaign_id)
    proposal["change_class"] = "parameter"
    proposal["change"]["parameter_surface"] = {
        "candidate": {"ggml_iqk": "1"},
        "anchor": {"ggml_iqk": "0"},
    }
    return proposal


def write_calibration_bundle(root: Path, *,
                             production_commit: str = campaign.PRODUCTION_COMMIT,
                             measurement_commit: str = campaign.MEASUREMENT_COMMIT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = {
        "schema": "epyc.autokernel.runtime_source_label.v1",
        "production_source_commit": production_commit,
        "measurement_instrument_commit": measurement_commit,
        "measurement_binary_sha256": "1" * 64,
        "copied_binary_sha256": "1" * 64,
        "measurement_linkage_sha256": "2" * 64,
        "measurement_toolchain_manifest_sha256": "4" * 64,
        "copied_linkage_sha256": "2" * 64,
        "binary_copy_exact": True,
    }
    source_sha = schemas.content_hash(source)
    (root / "runtime-source-label.json").write_text(
        json.dumps({**source, "source_sha256": source_sha}), encoding="utf-8")
    declaration = {
        "schema": "epyc.autokernel.live_control_campaign_declaration.v1",
        "campaign_id": "ak-controls-current-test",
        "recipe_id": campaign.HISTORICAL_CALIBRATED_RECIPE_ID,
        "contribution_floor": 0.03,
        "max_blocks_per_candidate": 20,
        "source_sha256": source_sha,
    }
    (root / "campaign_declaration.json").write_text(
        json.dumps(declaration), encoding="utf-8")
    summary = {
        "campaign_id": declaration["campaign_id"],
        "state": "controls_complete",
        "may_rank": True,
        "binary_copy_exact": True,
        "production_source_commit": production_commit,
        "calibration": {
            "outputs": {
                "accepted": True,
                "b_min_blocks": 12,
                "noise_floor_phi": 0.049206882811302755,
            },
            "attempts": [{
                "accepted": True,
                "mde": {"found": True, "value": 0.027408174371940427},
            }],
        },
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def physical_envelope(model: str = MODEL, frame_params=None, **overrides
                      ) -> physical_bounds.PhysicalEnvelope:
    frame_spec = spec(model=model)
    exact_params = frame_spec.bench_params if frame_params is None else frame_params
    kwargs = dict(
        shape_id=f"{campaign.DEFAULT_RECIPE_BY_BACKEND[campaign.BACKEND_CPU]}:{model}",
        delivered_unit="token",
        flops_per_unit=1.0,
        bytes_per_unit=1.0,
        peak_compute_flops_s=1e15,
        peak_memory_bytes_s=1e15,
        measurement_frame_sha256=physical_bounds.measurement_frame_sha256(
            frame_spec.recipe_id, exact_params),
        work_derivation_ref="test fixture: conservative work floor",
        hardware_peak_ref="test fixture: permissive peak ceiling",
    )
    kwargs.update(overrides)
    return physical_bounds.PhysicalEnvelope(**kwargs)


def ranked_units() -> tuple[campaign.RankedUnitSpec, ...]:
    return (
        campaign.RankedUnitSpec(
            unit_id="normal-prefill-512", kind=campaign.RANKED_UNIT_NORMAL,
            params={"n_prompt": 512},
            physical_envelope=physical_envelope(
                shape_id="normal-prefill-512",
                frame_params={**spec().bench_params, "n_prompt": 512})),
        campaign.RankedUnitSpec(
            unit_id="hard-prefill-511",
            kind=campaign.RANKED_UNIT_ANTI_SHORT_CIRCUIT,
            params={"n_prompt": 511},
            physical_envelope=physical_envelope(
                shape_id="hard-prefill-511",
                frame_params={**spec().bench_params, "n_prompt": 511})),
    )


# =============================================================================
# Fixtures built from the MEASURED A/A, not from an imagined noise model
# =============================================================================

def drifting(start: float, end: float, positions: int) -> tuple:
    """`positions` readings on the straight line the A/A actually walked.

    The A/A produced four whole-run readings; a run of five paired blocks makes
    ten invocations. Interpolating the observed endpoints across those ten
    positions is the honest way to put the MEASURED drift into a block-level
    fixture — it reproduces the direction, the magnitude and the monotonicity,
    which are the three properties that decided the design.
    """
    step = (end - start) / (positions - 1)
    return tuple(start + i * step for i in range(positions))


TG128_OVER_TEN_POSITIONS = drifting(campaign.AA_TG128_RUNS[0],
                                    campaign.AA_TG128_RUNS[-1], 10)


def pairs_from_positions(positions: tuple, *, candidate_factor: float = 1.0,
                         orders: tuple = ()) -> tuple:
    """Lay `positions` down as blocks of two, applying a true candidate effect.

    `orders` says which arm ran FIRST in each block. That is the whole point of
    the fixture: with `anchor_first` everywhere, the candidate always draws the
    later — and by measurement, slower — slot, which is exactly the systematic
    penalty a sequential A/B pays.
    """
    blocks = len(positions) // 2
    orders = orders or ("anchor_first",) * blocks
    out = []
    for i in range(blocks):
        first, second = positions[2 * i], positions[2 * i + 1]
        if orders[i] == "anchor_first":
            anchor, candidate = first, second * candidate_factor
        else:
            candidate, anchor = first * candidate_factor, second
        out.append(campaign.Pair(block_index=i, anchor=anchor, candidate=candidate,
                                 order=orders[i]))
    return tuple(out)


BALANCED = ("anchor_first", "candidate_first", "anchor_first", "candidate_first",
            "anchor_first")

PASSING_T0 = campaign.T0Outcome(all_pass=True, gates=(
    ("t0.everything", schemas.PASS, ()),))
FAILING_T0 = campaign.T0Outcome(all_pass=False, gates=(
    ("t0.correctness.backend_op_units", schemas.FAIL,
     ("MUL_MAT_ID produced a wrong result",)),))


# =============================================================================
# A recording spy for the loop's ORDER and its failure paths
# =============================================================================

class SpyOps:
    """Records every call; fails whichever stage it was told to fail.

    `executes=True` so the loop takes the branches an executing run takes —
    which is what makes the T0 short-circuit testable at all.
    """

    executes = True

    def __init__(self, *, fail_at: str = "", t0: campaign.T0Outcome = PASSING_T0,
                 pairs=None, release_raises: bool = False,
                 preflight_outcome: str = schemas.PASS) -> None:
        self.calls: list = []
        self.steps: tuple = ()
        self._fail_at = fail_at
        self._t0 = t0
        self._pairs = pairs
        self._release_raises = release_raises
        self._preflight_outcome = preflight_outcome

    def _record(self, name: str):
        self.calls.append(name)
        if self._fail_at == name:
            raise RuntimeError(f"induced failure at {name}")

    def record_proposal(self, spec_):
        self._record("record_proposal")

    def preflight(self, spec_):
        self._record("preflight")
        return schemas.Check(self._preflight_outcome, ("spy",))

    def acquire_claim(self, spec_):
        self._record("acquire_claim")
        return "claim"

    def release_claim(self, claim):
        self.calls.append("release_claim")
        if self._release_raises:
            raise RuntimeError("the flock could not be released")
        return "released"

    def create_worktree(self, spec_):
        self._record("create_worktree")
        return "worktree"

    def apply_candidate(self, spec_, tree):
        self._record("apply_candidate")

    def build(self, spec_, tree):
        self._record("build")
        return "build"

    def run_t0(self, spec_, build):
        self._record("run_t0")
        return self._t0

    def run_paired_blocks(self, spec_, build, claim):
        self._record("run_paired_blocks")
        return self._pairs

    def teardown_worktree(self, spec_, tree):
        self.calls.append("teardown_worktree")
        return "torn down"

    def keep_or_revert(self, spec_, tree, decision):
        self.calls.append("keep_or_revert")

    def prove_production_unchanged(self, spec_):
        self.calls.append("prove_production_unchanged")
        return schemas.Check(schemas.PASS, ("spy",))

    def journal(self, spec_, payload):
        self.calls.append("journal")
        self.journaled = dict(payload)


class EvaluationSpyOps(SpyOps):
    """The executing durability seam, with controllable evaluation failure."""

    def __init__(self, *args, evaluation_raises=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluation_raises = evaluation_raises

    def journal_evaluation(self, spec_, result):
        self.calls.append("journal_evaluation")
        if self._evaluation_raises:
            raise RuntimeError("induced evaluation append failure")

    def close_evaluation_window(self, spec_, tree):
        self._record("close_evaluation_window")


# =============================================================================
# 1. The dry run composes, end to end, and emits no number
# =============================================================================

class TestTheDryRunComposesEndToEnd(unittest.TestCase):

    def compose(self, **overrides):
        out = io.StringIO()
        ops = campaign.DryRunOps(out=out)
        result = campaign.run_campaign(spec(**overrides), ops)
        return result, ops, out.getvalue()

    def test_every_step_of_the_loop_is_composed(self):
        result, ops, _text = self.compose()
        self.assertEqual(result.state, campaign.STATE_COMPOSED)
        self.assertEqual(
            ops.calls,
            ["preflight", "acquire_claim", "create_worktree", "apply_candidate",
             "build", "t0", "paired_blocks", "keep_or_revert", "teardown_worktree",
             "release_claim", "prove_production_unchanged", "journal"])

    def test_the_dry_run_emits_no_speed_number(self):
        """A dry run that produces a number is a real run with a flag on it."""
        result, _ops, text = self.compose()
        self.assertEqual(result.pairs, ())
        self.assertIsNone(result.decision)
        self.assertIsNone(result.to_dict()["decision"])
        self.assertNotIn("tokens per second", text)

    def test_a_supplied_proposal_is_recorded_before_preflight(self):
        result, ops, _text = self.compose(proposal=proposal_manifest())
        self.assertEqual(ops.calls[0:2], ["record_proposal", "preflight"])
        self.assertEqual(result.to_dict()["spec"]["proposal"]["proposal_id"], "akp-test-0001")

    def test_the_composed_argv_is_the_canonical_recipe(self):
        """The argv is what drifted to 46% of canonical when nobody reviewed it."""
        _result, _ops, text = self.compose()
        self.assertIn("taskset -c 0-95 numactl --interleave=all", text)
        self.assertIn("-t 96 -fa 1 -mmp 0", text)
        for name in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_DYNAMIC",
                     "GGML_IQK"):
            self.assertIn(name, text)

    def test_both_arms_differ_only_in_the_binary(self):
        rendered = campaign.render_bench_commands(spec())
        anchor, candidate = rendered["anchor"]["argv"], rendered["candidate"]["argv"]
        self.assertEqual(len(anchor), len(candidate))
        differing = [i for i, (a, c) in enumerate(zip(anchor, candidate)) if a != c]
        self.assertEqual(len(differing), 1, f"arms differ at {differing}")
        self.assertIn("llama-bench", anchor[differing[0]])

    def test_construct_candidate_uses_external_build_binding(self):
        ops = campaign.HostOps.__new__(campaign.HostOps)
        with tempfile.TemporaryDirectory() as root:
            snapshot_root = Path(root) / "snapshot"
            snapshot_root.joinpath(".git").mkdir(parents=True)
            build_root = Path(root) / "build-clean"
            snapshot = mock.Mock(path=mock.Mock(path=str(snapshot_root)))
            plan = mock.Mock(build_dir=mock.Mock(path=str(build_root)))
            ops._build_state = {"plan": plan, "tree": snapshot}
            campaign_spec = spec()
            with mock.patch.object(campaign.recipes, "construct",
                                   return_value=mock.sentinel.command) as construct:
                result = ops._construct(campaign_spec, arm="candidate")
        self.assertIs(result, mock.sentinel.command)
        binding = construct.call_args.kwargs["binding"]
        self.assertTrue(binding.external_build_root.endswith("build-clean"))
        self.assertTrue(binding.source_root.endswith("snapshot"))
        self.assertTrue(binding.binary.endswith("build-clean/bin/llama-bench"))

    def test_construct_anchor_uses_git_source_and_external_build_binding(self):
        ops = campaign.HostOps.__new__(campaign.HostOps)
        ops._build_state = {"plan": mock.Mock(), "tree": mock.Mock(
            path=mock.Mock(path="/snapshot"))}
        campaign_spec = spec()
        with mock.patch.object(campaign.recipes, "construct",
                               return_value=mock.sentinel.command) as construct:
            ops._construct(campaign_spec, arm="anchor")
        binding = construct.call_args.kwargs["binding"]
        self.assertEqual(binding.source_root, campaign.MEASUREMENT_REPO)
        self.assertEqual(binding.external_build_root, campaign.MEASUREMENT_BUILD_ROOT)
        self.assertTrue(binding.binary.startswith(campaign.MEASUREMENT_BUILD_ROOT + "/"))
        self.assertTrue(binding.library_path.startswith(campaign.MEASUREMENT_BUILD_ROOT + "/"))

    def test_parameter_proposal_renders_the_iqk_difference_on_the_two_arms(self):
        rendered = campaign.render_bench_commands(
            spec(proposal=iqk_parameter_proposal()))
        self.assertEqual(rendered["candidate"]["env"]["GGML_IQK"], "1")
        self.assertEqual(rendered["anchor"]["env"]["GGML_IQK"], "0")

    def test_serving_and_measurement_anchors_are_distinct_and_v9_pinned(self):
        rendered = campaign.render_bench_commands(spec())
        anchor_binary = rendered["anchor"]["argv"][
            next(i for i, value in enumerate(rendered["anchor"]["argv"])
                 if value.endswith("/llama-bench"))
        ]
        self.assertTrue(anchor_binary.startswith(campaign.MEASUREMENT_BUILD_ROOT + "/"))
        payload = spec().to_dict()
        self.assertEqual(payload["anchor"]["expected_commit"], campaign.PRODUCTION_COMMIT)
        self.assertEqual(payload["measurement_instrument"]["expected_commit"],
                         campaign.MEASUREMENT_COMMIT)
        self.assertNotEqual(payload["anchor"]["repo"],
                            payload["measurement_instrument"]["repo"])
        from .execution import live_controls
        self.assertEqual(live_controls.PRODUCTION_COMMIT, campaign.PRODUCTION_COMMIT)
        self.assertEqual(live_controls.INSTRUMENT_BRANCH, campaign.MEASUREMENT_BRANCH)
        self.assertEqual(live_controls.INSTRUMENT_COMMIT, campaign.MEASUREMENT_COMMIT)
        self.assertEqual(campaign.MEASUREMENT_COMMIT,
                         "283b520b527a7b507d6cf05cd124a59f427f3629")

    def test_the_ledger_released_everything(self):
        result, _ops, _text = self.compose()
        self.assertTrue(all(r.released for r in result.releases))
        self.assertEqual([r.name for r in result.releases],
                         ["campaign_worktree", "cpu_region_claim"])

    def test_composition_and_execution_walk_the_same_steps(self):
        """A second spelling of the loop is the defect chain.py warns about.

        The composition pass and an executing pass must call the same ops in the
        same order; the only licensed difference is the T0 short-circuit, which
        a composition pass cannot take because it executed no T0.
        """
        _result, dry, _text = self.compose()
        spy = SpyOps(pairs=pairs_from_positions(TG128_OVER_TEN_POSITIONS,
                                                candidate_factor=1.08, orders=BALANCED))
        campaign.run_campaign(spec(), spy)
        rename = {"t0": "run_t0", "paired_blocks": "run_paired_blocks"}
        self.assertEqual([rename.get(c, c) for c in dry.calls], spy.calls)


# =============================================================================
# 2. Dry run is the DEFAULT
# =============================================================================

class TestDryRunIsTheDefault(unittest.TestCase):

    def test_the_parser_defaults_to_dry_run(self):
        args = campaign.build_parser().parse_args(["--model", MODEL])
        self.assertTrue(args.dry_run)

    def test_execute_without_the_host_attestation_refuses(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err):
            code = campaign.main(["--model", MODEL, "--execute"], out=out)
        self.assertEqual(code, 2)
        self.assertIn("--i-hold-the-host", err.getvalue())

    def test_execute_with_the_attestation_is_accepted_by_the_parser(self):
        """The compliant-path control: the guard must not forbid its own idiom."""
        args = campaign.build_parser().parse_args(
            ["--model", MODEL, "--execute", "--i-hold-the-host"])
        self.assertFalse(args.dry_run)
        self.assertTrue(args.i_hold_the_host)

    def test_main_runs_the_composition_and_exits_zero(self):
        out = io.StringIO()
        code = campaign.main(["--model", MODEL], out=out,
                             ops=campaign.DryRunOps(out=out))
        self.assertEqual(code, 0)
        self.assertIn("dry_run_composed", out.getvalue())

    def test_json_mode_emits_one_parseable_document_on_the_output_stream(self):
        out, detail = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(detail):
            code = campaign.main(["--model", MODEL, "--json"], out=out)
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["state"], "dry_run_composed")
        self.assertFalse(payload["executed"])
        self.assertTrue(out.getvalue().lstrip().startswith("{"))
        self.assertIn("DRY RUN", detail.getvalue())

    def test_main_refuses_a_bad_spec_before_anything_starts(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = campaign.main(["--model", MODEL, "--campaign-id", "nope"],
                                 out=io.StringIO())
        self.assertEqual(code, 2)


# =============================================================================
# 3. Every failure path releases the claim and tears down the worktree
# =============================================================================

def _raising(exc, ops, name):
    def stage(*_args, **_kwargs):
        ops.calls.append(name)
        raise exc
    return stage


class TestEveryFailurePathReleases(unittest.TestCase):

    STAGES_AFTER_THE_CLAIM = ("create_worktree", "apply_candidate", "build", "run_t0",
                              "run_paired_blocks")

    def test_a_failure_at_any_stage_releases_the_claim(self):
        for stage in self.STAGES_AFTER_THE_CLAIM:
            with self.subTest(stage=stage):
                ops = SpyOps(fail_at=stage)
                result = campaign.run_campaign(spec(), ops)
                self.assertEqual(result.state, campaign.STATE_ERROR)
                self.assertIn("release_claim", ops.calls)
                released = {r.name: r.released for r in result.releases}
                self.assertTrue(released.get("cpu_region_claim"),
                                f"{stage}: the claim was not released")

    def test_a_failure_after_the_worktree_exists_tears_it_down(self):
        for stage in ("apply_candidate", "build", "run_t0", "run_paired_blocks"):
            with self.subTest(stage=stage):
                ops = SpyOps(fail_at=stage)
                campaign.run_campaign(spec(), ops)
                self.assertIn("teardown_worktree", ops.calls)
                self.assertLess(ops.calls.index("teardown_worktree"),
                                ops.calls.index("release_claim"),
                                "the worktree must be torn down INSIDE the claim window")

    def test_a_failure_before_the_claim_acquires_nothing_to_leak(self):
        ops = SpyOps(fail_at="acquire_claim")
        result = campaign.run_campaign(spec(), ops)
        self.assertEqual(result.state, campaign.STATE_ERROR)
        self.assertEqual(result.releases, ())
        self.assertNotIn("release_claim", ops.calls)

    def test_the_production_proof_runs_on_the_failing_path_too(self):
        """The check that mattered must not be skipped by the failure that made it matter."""
        for stage in self.STAGES_AFTER_THE_CLAIM:
            with self.subTest(stage=stage):
                ops = SpyOps(fail_at=stage)
                result = campaign.run_campaign(spec(), ops)
                self.assertIn("prove_production_unchanged", ops.calls)
                self.assertIsNotNone(result.production_unchanged)

    def test_a_refused_preflight_stops_before_the_claim(self):
        ops = SpyOps(preflight_outcome=schemas.FAIL)
        result = campaign.run_campaign(spec(), ops)
        self.assertEqual(result.state, campaign.STATE_PREFLIGHT_REFUSED)
        self.assertNotIn("acquire_claim", ops.calls)

    def test_a_keyboard_interrupt_still_releases_the_claim(self):
        """The realistic early exit: Ctrl-C an hour into a claim window.

        `KeyboardInterrupt` is not an `Exception`, so a driver that caught only
        `Exception` would leave the one interruption an operator actually
        performs as the one path that leaks the claim.
        """
        ops = SpyOps()
        ops.build = _raising(KeyboardInterrupt("operator pressed Ctrl-C"), ops, "build")
        result = campaign.run_campaign(spec(), ops)
        self.assertEqual(result.state, campaign.STATE_ERROR)
        self.assertIn("release_claim", ops.calls)
        self.assertTrue(all(r.released for r in result.releases))

    def test_a_release_that_raises_is_recorded_not_swallowed(self):
        ops = SpyOps(fail_at="build", release_raises=True)
        result = campaign.run_campaign(spec(), ops)
        released = {r.name: r.released for r in result.releases}
        self.assertFalse(released["cpu_region_claim"])
        self.assertFalse(result.ok, "a leaked claim must not report a clean campaign")
        # ... and the OTHER release still ran.
        self.assertTrue(released["campaign_worktree"])

    def test_a_failing_production_proof_makes_the_campaign_not_ok(self):
        ops = SpyOps(pairs=pairs_from_positions(TG128_OVER_TEN_POSITIONS,
                                                candidate_factor=1.08, orders=BALANCED))
        ops.prove_production_unchanged = lambda _s: schemas.Check(
            schemas.FAIL, ("llama.cpp moved",))
        result = campaign.run_campaign(spec(), ops)
        self.assertFalse(result.ok)


class TestTheResourceLedger(unittest.TestCase):

    def test_it_releases_in_reverse_order(self):
        seen: list = []
        ledger = campaign.ResourceLedger()
        ledger.hold("first", lambda: seen.append("first"))
        ledger.hold("second", lambda: seen.append("second"))
        ledger.release_all()
        self.assertEqual(seen, ["second", "first"])

    def test_it_is_idempotent(self):
        seen: list = []
        ledger = campaign.ResourceLedger()
        ledger.hold("only", lambda: seen.append("only"))
        first = ledger.release_all()
        second = ledger.release_all()
        self.assertEqual(seen, ["only"])
        self.assertIs(first, second)

    def test_a_failing_release_does_not_strand_the_rest(self):
        seen: list = []

        def boom():
            raise OSError("flock is gone")

        ledger = campaign.ResourceLedger()
        ledger.hold("claim", lambda: seen.append("claim"))
        ledger.hold("worktree", boom)
        records = {r.name: r for r in ledger.release_all()}
        self.assertEqual(seen, ["claim"], "the claim must still be released")
        self.assertFalse(records["worktree"].released)
        self.assertIn("flock is gone", records["worktree"].detail)

    def test_registering_after_release_is_refused(self):
        ledger = campaign.ResourceLedger()
        ledger.release_all()
        with self.assertRaises(RuntimeError):
            ledger.hold("late", lambda: None)


# =============================================================================
# 4. A failed T0 computes no speed number AT ALL
# =============================================================================

class TestAFailedT0ComputesNoSpeedNumber(unittest.TestCase):

    def test_the_blocks_are_never_run(self):
        ops = SpyOps(t0=FAILING_T0)
        result = campaign.run_campaign(spec(), ops)
        self.assertEqual(result.state, campaign.STATE_T0_FAILED)
        self.assertNotIn("run_paired_blocks", ops.calls,
                         "a wrong kernel got as far as a benchmark")
        self.assertEqual(result.pairs, ())
        self.assertIsNone(result.decision)

    def test_the_claim_is_still_released_and_production_still_proved(self):
        ops = SpyOps(t0=FAILING_T0)
        result = campaign.run_campaign(spec(), ops)
        self.assertTrue(all(r.released for r in result.releases))
        self.assertIn("prove_production_unchanged", ops.calls)

    def test_the_rule_itself_refuses_to_rank_a_wrong_kernel(self):
        """The second lock on the same door: ordering alone is a convention."""
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.5,
                                     orders=BALANCED)
        with self.assertRaises(campaign.AcceptRuleMisuse):
            campaign.decide(pairs, t0=FAILING_T0, blocks_precommitted=5,
                            drift_bound=0.02)

    def test_the_compliant_path_still_ranks(self):
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.5,
                                     orders=BALANCED)
        decision = campaign.decide(pairs, t0=PASSING_T0, blocks_precommitted=5,
                                   drift_bound=0.02)
        self.assertTrue(decision.keep)

    def test_an_ops_that_declares_itself_a_dry_run_still_cannot_rank_a_wrong_kernel(self):
        """The T0 short-circuit is guarded by `ops.executes`, which is the ops
        object's own word. An object that says `executes = False` and returns
        real blocks anyway walks straight past the short-circuit — and is then
        stopped by the rule itself, which is why the second lock exists.

        The campaign ends in ERROR with NO decision and NO speed rank.
        """
        class Lying(SpyOps):
            executes = False

        ops = Lying(t0=FAILING_T0,
                    pairs=pairs_from_positions(TG128_OVER_TEN_POSITIONS,
                                               candidate_factor=1.5, orders=BALANCED))
        result = campaign.run_campaign(spec(), ops)
        self.assertEqual(result.state, campaign.STATE_ERROR)
        self.assertIsNone(result.decision)
        self.assertIn("AcceptRuleMisuse", result.error)
        self.assertTrue(all(r.released for r in result.releases))


# =============================================================================
# 5. The accept rule, against the measured A/A
# =============================================================================

class TestTheAcceptRule(unittest.TestCase):

    BOUND = campaign.DRIFT_BOUND_BY_METRIC["decode_tokens_per_s"]

    def decide(self, pairs, **kw):
        kwargs = dict(t0=PASSING_T0, blocks_precommitted=5, drift_bound=self.BOUND)
        kwargs.update(kw)
        return campaign.decide(pairs, **kwargs)

    # -- the null ---------------------------------------------------------

    def test_the_measured_AA_drift_alone_is_rejected(self):
        """The null fixture is the A/A itself: identical code, real drift."""
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.0,
                                     orders=BALANCED)
        decision = self.decide(pairs)
        self.assertFalse(decision.keep)
        self.assertIn("REVERT", decision.reason)

    def test_a_sequential_design_would_have_charged_the_candidate_for_the_drift(self):
        """Anchor-first EVERYWHERE: the candidate always draws the later, slower slot.

        This is the measurement that made interleaving mandatory, expressed as a
        test: with identical code, every pair reads against the candidate.
        """
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.0)
        self.assertTrue(all(p.delta < 0 for p in pairs))
        decision = self.decide(pairs)
        self.assertFalse(decision.keep)
        self.assertLess(decision.min_delta, 0)

    def test_a_marginal_win_inside_the_measured_noise_is_rejected(self):
        """+1.5% on a host whose own A/A spread is 4.3% is not resolvable."""
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.015,
                                     orders=BALANCED)
        decision = self.decide(pairs)
        self.assertFalse(decision.keep)
        self.assertGreater(decision.min_delta, 0, "the sign test alone would have passed it")
        self.assertIn("contribution floor", decision.reason)

    def test_one_adverse_block_sinks_an_otherwise_positive_candidate(self):
        pairs = list(pairs_from_positions(TG128_OVER_TEN_POSITIONS,
                                          candidate_factor=1.08, orders=BALANCED))
        good = pairs[2]
        pairs[2] = campaign.Pair(block_index=good.block_index, anchor=good.anchor,
                                 candidate=good.anchor * 0.999, order=good.order)
        decision = self.decide(tuple(pairs))
        self.assertFalse(decision.keep)
        self.assertIn("did not favour", decision.reason)

    # -- the win ----------------------------------------------------------

    def test_a_real_win_is_accepted(self):
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.08,
                                     orders=BALANCED)
        decision = self.decide(pairs)
        self.assertTrue(decision.keep, decision.reason)
        self.assertGreater(decision.min_delta, 0)
        self.assertGreater(decision.median_relative, self.BOUND)
        self.assertIn("KEEP", decision.reason)

    def test_a_win_survives_the_drift_it_was_measured_through(self):
        """Same true effect, under the worst ADMISSIBLE order draw: 4-1.

        The interleaving does not remove the drift, it bounds it to one step;
        a real effect must therefore still clear the bound under the most
        lopsided draw the order control still admits, and this is that
        assertion. (5-0 is a different thing — a sequential A/B — and is
        refused outright by `TestTheOrderDrawMustNotBeDegenerate`.)
        """
        lopsided = ("anchor_first",) * 4 + ("candidate_first",)
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.08,
                                     orders=lopsided)
        decision = self.decide(pairs)
        self.assertTrue(decision.keep, decision.reason)

    # -- the neutral control ----------------------------------------------

    def test_a_run_whose_anchor_arm_moved_is_inadmissible(self):
        """THE BITE. Interleaving hides drift; it does not detect it.

        The candidate here is a genuine +8%, so both accept conjuncts pass —
        and the run is still refused, because the anchor arm (identical code)
        slid further across the run than an A/A of identical code ever did.
        Without this, "this kernel is faster" and "this kernel ran first" are
        the same record.
        """
        collapsing = drifting(52.76, 39.14, 10)   # the retracted E/F magnitude
        pairs = pairs_from_positions(collapsing, candidate_factor=1.08, orders=BALANCED)
        decision = self.decide(pairs)
        self.assertFalse(decision.keep)
        self.assertIn("inadmissible", decision.reason)
        self.assertGreater(decision.anchor_drift, self.BOUND)
        # ... and it was NOT refused for lack of a candidate effect.
        self.assertGreater(decision.min_delta, 0)
        self.assertGreater(decision.median_relative, self.BOUND)

    def test_a_stable_run_is_admissible(self):
        """Compliant-path control: the measured A/A drift itself is within bound."""
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.08,
                                     orders=BALANCED)
        decision = self.decide(pairs)
        self.assertLessEqual(decision.anchor_drift, self.BOUND)
        self.assertTrue(decision.keep, decision.reason)

    def test_the_control_reads_the_anchor_arm_the_run_already_produced(self):
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.08,
                                     orders=BALANCED)
        self.assertEqual(campaign.anchor_drift(pairs),
                         max(abs(s) for s in campaign.adjacent_relative_steps(
                             [p.anchor for p in pairs])))

    def test_the_control_is_evaluated_in_block_order_not_argument_order(self):
        """A shuffled sequence must not be able to flatten the drift it contains."""
        pairs = pairs_from_positions(drifting(52.76, 39.14, 10), candidate_factor=1.08,
                                     orders=BALANCED)
        shuffled = (pairs[2], pairs[0], pairs[4], pairs[1], pairs[3])
        self.assertAlmostEqual(campaign.anchor_drift(shuffled),
                               campaign.anchor_drift(pairs))
        self.assertFalse(self.decide(shuffled).keep)

    # -- no optional stopping ---------------------------------------------

    def test_more_blocks_than_precommitted_are_refused(self):
        """§6.5's re-run-until-it-crosses hole has nothing to re-run."""
        pairs = pairs_from_positions(drifting(52.76, 50.52, 12), candidate_factor=1.08)
        self.assertEqual(len(pairs), 6)
        with self.assertRaises(campaign.AcceptRuleMisuse) as raised:
            self.decide(pairs)
        self.assertIn("optional stopping", str(raised.exception))

    def test_fewer_blocks_than_precommitted_are_refused(self):
        pairs = pairs_from_positions(drifting(52.76, 50.52, 8), candidate_factor=1.08)
        with self.assertRaises(campaign.AcceptRuleMisuse):
            self.decide(pairs)

    def test_exactly_the_precommitted_count_is_accepted(self):
        """Compliant-path control for the two refusals above."""
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.08,
                                     orders=BALANCED)
        self.assertEqual(len(pairs), 5)
        self.assertTrue(self.decide(pairs).keep)

    # -- the record --------------------------------------------------------

    def test_the_decision_publishes_every_number_it_read(self):
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.08,
                                     orders=BALANCED)
        payload = self.decide(pairs).to_dict()
        for key in ("keep", "reason", "blocks", "min_delta", "median_relative",
                    "drift_bound", "deltas", "relatives"):
            self.assertIn(key, payload)
        self.assertEqual(len(payload["deltas"]), 5)

    def test_a_zero_or_negative_reading_is_not_a_measurement(self):
        with self.assertRaises(ValueError):
            campaign.Pair(block_index=0, anchor=0.0, candidate=1.0, order="anchor_first")

    def test_a_block_must_name_an_order_the_planner_can_derive(self):
        with self.assertRaises(ValueError):
            campaign.Pair(block_index=0, anchor=1.0, candidate=1.0, order="whatever")


class TestTheDriftBoundIsDerivedFromTheMeasurement(unittest.TestCase):

    def test_it_is_the_largest_adjacent_step_in_the_recorded_series(self):
        steps = campaign.adjacent_relative_steps(campaign.AA_TG128_RUNS)
        self.assertEqual(len(steps), 3)
        self.assertAlmostEqual(campaign.drift_bound_from(campaign.AA_TG128_RUNS),
                               max(abs(s) for s in steps))

    def test_the_recorded_series_are_the_ones_that_were_measured(self):
        """If the A/A is re-run, these change and the rule changes with them."""
        self.assertEqual(campaign.AA_PP512_RUNS, (899.95, 894.70, 867.16, 886.16))
        self.assertEqual(campaign.AA_TG128_RUNS, (52.76, 52.31, 51.62, 50.52))

    def test_decode_declined_monotonically(self):
        """The fact that made interleaved paired blocks mandatory."""
        runs = campaign.AA_TG128_RUNS
        self.assertTrue(all(runs[i + 1] < runs[i] for i in range(len(runs) - 1)))

    def test_the_bounds_bracket_the_measured_values(self):
        self.assertAlmostEqual(campaign.DRIFT_BOUND_BY_METRIC["prefill_tokens_per_s"],
                               0.03077, places=4)
        self.assertAlmostEqual(campaign.DRIFT_BOUND_BY_METRIC["decode_tokens_per_s"],
                               0.02131, places=4)

    def test_an_unmeasured_cell_gets_the_most_conservative_bound(self):
        self.assertEqual(campaign.drift_bound_for_metric("op_throughput_gflops"),
                         max(campaign.DRIFT_BOUND_BY_METRIC.values()))

    def test_a_single_observation_cannot_produce_a_bound(self):
        with self.assertRaises(ValueError):
            campaign.drift_bound_from((52.76,))


class TestTheAcceptedCalibrationBindsTheLiveRule(unittest.TestCase):

    def test_the_v8_constants_are_historical_regression_fixtures(self):
        repo = Path(campaign.__file__).resolve().parents[3]
        summary = json.loads((
            repo / campaign.HISTORICAL_CALIBRATION_EVIDENCE_REF / "summary.json"
        ).read_text(encoding="utf-8"))
        declaration = json.loads((
            repo / campaign.HISTORICAL_CALIBRATION_EVIDENCE_REF / "campaign_declaration.json"
        ).read_text(encoding="utf-8"))
        outputs = summary["calibration"]["outputs"]
        attempt = summary["calibration"]["attempts"][0]
        self.assertTrue(outputs["accepted"])
        self.assertEqual(campaign.HISTORICAL_CALIBRATED_RECIPE_ID,
                         declaration["recipe_id"])
        self.assertEqual(campaign.HISTORICAL_CONTRIBUTION_FLOOR,
                         declaration["contribution_floor"])
        self.assertEqual(campaign.HISTORICAL_B_MIN_BLOCKS,
                         outputs["b_min_blocks"])
        self.assertEqual(campaign.HISTORICAL_MAX_BLOCKS,
                         declaration["max_blocks_per_candidate"])
        self.assertEqual(campaign.HISTORICAL_NOISE_FLOOR_PHI,
                         outputs["noise_floor_phi"])
        self.assertEqual(campaign.HISTORICAL_MDE, attempt["mde"]["value"])
        with self.assertRaisesRegex(ValueError, "stale|absent"):
            campaign.load_calibration_bundle(
                repo / campaign.HISTORICAL_CALIBRATION_EVIDENCE_REF)

    def test_a_current_bundle_supplies_the_cli_recipe_floor_and_b_min(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = write_calibration_bundle(Path(tmp) / "calibration")
            out = io.StringIO()
            self.assertEqual(campaign.main(
                ["--model", MODEL, "--calibration-bundle", str(bundle)], out=out), 0)
            rendered = out.getvalue()
            self.assertIn(campaign.HISTORICAL_CALIBRATED_RECIPE_ID, rendered)
            self.assertIn("over 12 pre-committed pairs", rendered)
            self.assertIn("median(relative) > 3.0000%", rendered)

    def test_an_uncalibrated_cell_has_no_live_ranking_authority(self):
        built = spec(recipe_id="t1b.llama_cpu.llama_bench_decode.v1", blocks=12)
        check = campaign.HostOps().calibration_gate(built)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("cell-local", " ".join(check.reasons))

    def test_the_contribution_floor_is_not_the_anchor_movement_bound(self):
        pairs = pairs_from_positions(
            (100.0,) * 10, candidate_factor=1.025, orders=BALANCED)
        decision = campaign.decide(
            pairs, t0=PASSING_T0, blocks_precommitted=5,
            drift_bound=0.02, contribution_floor=0.03)
        self.assertFalse(decision.keep)
        self.assertLessEqual(decision.anchor_drift, 0.02)
        self.assertGreater(decision.median_relative, decision.drift_bound)
        self.assertLess(decision.median_relative, decision.contribution_floor)
        self.assertEqual(decision.to_dict()["contribution_floor"], 0.03)


# =============================================================================
# 6. The idle-frequency trap
# =============================================================================

class TestTheBoostCheckIsOnlyEvaluatedUnderLoad(unittest.TestCase):
    """Bite, control, and the guard still biting — all three, on real readings.

    The numbers are this host's, measured 2026-08-04: 16 cores above 2.5 GHz at
    idle, 117 under load, 35 while another session's stack was coming up.
    """

    def test_an_idle_healthy_host_is_not_a_failure(self):
        """THE BITE. The check as written aborts a campaign on a good machine."""
        check = campaign.check_boost_under_load(boosting_cores=16, load1=3.3,
                                                cpu_count=96)
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("IDLE", " ".join(check.reasons))

    def test_it_is_not_a_pass_either(self):
        """COULD_NOT_CHECK is a third outcome. An unread throttle check is not clean."""
        check = campaign.check_boost_under_load(boosting_cores=16, load1=3.3,
                                                cpu_count=96)
        self.assertNotEqual(check.outcome, schemas.PASS)

    def test_a_healthy_loaded_host_passes(self):
        """THE COMPLIANT-PATH CONTROL: 117 boosting under our own bench.

        `-t 96` on the 96 claimed cores drives load to roughly 1.0/core, which
        is where the boost count means what the recipe says it means.
        """
        check = campaign.check_boost_under_load(boosting_cores=117, load1=96.0,
                                                cpu_count=96)
        self.assertEqual(check.outcome, schemas.PASS)

    def test_a_throttled_loaded_host_still_FAILS(self):
        """The guard must still bite, or fixing it is deleting it.

        `feedback_host_throttle_check`: this host sat at roughly 40% of its
        normal clock for days, undetected. Under our own bench that reads as a
        collapsed boost count, and it must FAIL rather than annotate.
        """
        check = campaign.check_boost_under_load(boosting_cores=35, load1=96.0,
                                                cpu_count=96)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("poisoned", " ".join(check.reasons))

    def test_a_co_tenant_stack_coming_up_is_not_a_verdict_on_our_clock(self):
        """The reading that voided A/A runs E and F: load 23.9, 35 boosting.

        That was another session's seven `llama-server` instances, not our
        bench. It is neither a healthy host nor a throttled one, and the check
        must decline to rule rather than blame the CPU for a co-tenant.
        """
        check = campaign.check_boost_under_load(boosting_cores=35, load1=23.9,
                                                cpu_count=96)
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_the_load_floor_is_the_packages_own_contention_ceiling(self):
        """Derived, not chosen: the same number microbench refuses to start above."""
        from .execution import microbench
        self.assertEqual(campaign.LOADED_ENOUGH_TO_JUDGE_BOOST,
                         microbench.HostStatePolicy().max_load_per_core)

    def test_an_unreadable_load_is_could_not_check(self):
        check = campaign.check_boost_under_load(boosting_cores=117, load1=None,
                                                cpu_count=96)
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_the_threshold_is_exactly_the_canonical_one(self):
        self.assertEqual(campaign.BOOST_THRESHOLD_KHZ, 2_500_000)
        self.assertEqual(campaign.BOOST_MIN_CORES, 80)
        at_threshold = campaign.check_boost_under_load(boosting_cores=80, load1=96.0,
                                                       cpu_count=96)
        below = campaign.check_boost_under_load(boosting_cores=79, load1=96.0,
                                                cpu_count=96)
        self.assertEqual(at_threshold.outcome, schemas.PASS)
        self.assertEqual(below.outcome, schemas.FAIL)

    def test_the_one_week_uptime_ceiling_refuses_without_rebooting(self):
        self.assertEqual(campaign.check_host_uptime(6.99 * 86400).outcome,
                         schemas.PASS)
        over = campaign.check_host_uptime(7.01 * 86400)
        self.assertEqual(over.outcome, schemas.FAIL)
        self.assertIn("reboot decision package", " ".join(over.reasons))
        self.assertEqual(campaign.check_host_uptime(None).outcome,
                         schemas.COULD_NOT_CHECK)


# =============================================================================
# 7. The MoE-dispatch hole
# =============================================================================

class TestTheOpSuiteMustCoverMoEDispatch(unittest.TestCase):

    def test_a_suite_without_MUL_MAT_ID_is_refused(self):
        """THE BITE. The predecessor harness tested MUL_MAT only."""
        with self.assertRaises(ValueError) as raised:
            campaign.require_op_suite_covers_moe_dispatch(("MUL_MAT",))
        self.assertIn("MUL_MAT_ID", str(raised.exception))

    def test_a_suite_with_it_is_accepted(self):
        """Compliant-path control."""
        self.assertEqual(
            campaign.require_op_suite_covers_moe_dispatch(("MUL_MAT", "MUL_MAT_ID")),
            ("MUL_MAT", "MUL_MAT_ID"))

    def test_the_campaign_spec_cannot_be_built_without_it(self):
        with self.assertRaises(ValueError):
            spec(t0_ops=("MUL_MAT",))

    def test_the_default_spec_covers_it(self):
        self.assertIn(campaign.MOE_DISPATCH_OP, spec().t0_ops)

    def test_an_empty_suite_is_refused(self):
        with self.assertRaises(ValueError):
            campaign.require_op_suite_covers_moe_dispatch(())


# =============================================================================
# 8. The spec is a PRE-COMMITMENT
# =============================================================================

class TestTheSpecIsAPreCommitment(unittest.TestCase):

    def test_it_is_frozen(self):
        with self.assertRaises(Exception):
            spec().blocks = 9

    def test_t0_suite_seed_is_deterministic_and_serialized(self):
        first = spec()
        same = spec()
        different = spec(candidate_id="akc-other")
        self.assertEqual(first.suite_seed, same.suite_seed)
        self.assertNotEqual(first.suite_seed, different.suite_seed)
        self.assertEqual(first.to_dict()["suite_seed"], first.suite_seed)

    def test_unmatched_campaign_seed_contract_is_byte_compatible(self):
        """Adding matched experiments must not silently reseed legacy campaigns."""
        built = spec(created_at="2026-08-12T00:00:00+00:00")
        self.assertEqual(built.bench_params["autokernel_seed"],
                         int("450223030" + "8888835942"))
        self.assertEqual(built.suite_seed, int("141681052" + "43381952088"))
        self.assertEqual(built.schedule_seed,
                         "ak-test/2026-08-12T00:00:00+00:00")
        self.assertEqual(built.holdout_selection_seed,
                         "ak-test/14168105243381952088")

    def test_proposal_v4_is_validated_and_frozen_by_value(self):
        proposal = proposal_manifest()
        built = spec(proposal=proposal)
        proposal["hypothesis"] = "mutated after validation"
        self.assertNotEqual(built.proposal["hypothesis"], proposal["hypothesis"])
        self.assertEqual(built.proposal_id, "akp-test-0001")

    def test_historical_proposal_v2_cannot_drive_a_new_execution(self):
        proposal = proposal_manifest()
        proposal["schema"] = schemas.SCHEMA_PROPOSAL_V2
        del proposal["representation_contract"]
        with self.assertRaisesRegex(ValueError, "proposal manifest is invalid"):
            spec(proposal=proposal)

    def test_proposal_cannot_cross_campaigns(self):
        with self.assertRaisesRegex(ValueError, "does not match campaign"):
            spec(proposal=proposal_manifest("ak-other"))

    def test_proposal_provider_cannot_target_a_different_backend(self):
        proposal = proposal_manifest()
        proposal["provider_reference"]["target_backend"] = campaign.BACKEND_GPU
        with self.assertRaisesRegex(ValueError, "provider target.*does not match"):
            spec(proposal=proposal)

    def test_proposal_provider_commit_must_match_measurement_instrument(self):
        proposal = proposal_manifest()
        proposal["provider_reference"]["source_commit"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "provider source commit.*measurement instrument"):
            spec(proposal=proposal)

    def test_proposal_provider_symlink_into_shared_rocm_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="ak-provider-link-") as root:
            link = Path(root, "rocm")
            link.symlink_to("/opt/rocm", target_is_directory=True)
            proposal = proposal_manifest()
            proposal["provider_reference"]["isolation_root"] = str(link)
            with self.assertRaisesRegex(ValueError, "provider isolation"):
                spec(proposal=proposal)

    def test_parameter_proposal_binds_the_registered_iqk_arm_variant(self):
        built = spec(proposal=iqk_parameter_proposal())
        self.assertEqual(built.candidate_param_overrides, {"ggml_iqk": "1"})
        self.assertEqual(built.anchor_param_overrides, {"ggml_iqk": "0"})
        self.assertEqual(built.params_for_arm("candidate")["ggml_iqk"], "1")
        self.assertEqual(built.params_for_arm("anchor")["ggml_iqk"], "0")
        self.assertEqual(built.t0_parameter_env_for_arm("candidate"),
                         (("GGML_IQK", "1"),))
        self.assertEqual(built.t0_parameter_env_for_arm("anchor"),
                         (("GGML_IQK", "0"),))

    def test_parameter_proposal_cannot_name_an_unregistered_or_null_variant(self):
        bad = iqk_parameter_proposal()
        bad["change"]["parameter_surface"]["candidate"] = {"other": "1"}
        with self.assertRaisesRegex(ValueError, "licenses only"):
            spec(proposal=bad)
        same = iqk_parameter_proposal()
        same["change"]["parameter_surface"]["anchor"] = {"ggml_iqk": "1"}
        with self.assertRaisesRegex(ValueError, "declares no comparison"):
            spec(proposal=same)

    def test_physical_envelope_is_bound_to_the_exact_campaign_unit(self):
        bound = spec(physical_envelope=physical_envelope())
        self.assertEqual(bound.to_dict()["physical_envelope"],
                         physical_envelope().to_dict())
        with self.assertRaisesRegex(ValueError, "does not match campaign unit"):
            spec(physical_envelope=physical_envelope(shape_id="an-easier-cell"))
        with self.assertRaisesRegex(ValueError, "registered llama-bench.*token/s"):
            spec(physical_envelope=physical_envelope(delivered_unit="output_row"))
        with self.assertRaisesRegex(ValueError, "exact recipe, model, and parameters"):
            spec(physical_envelope=physical_envelope(
                measurement_frame_sha256=physical_bounds.measurement_frame_sha256(
                    campaign.DEFAULT_RECIPE_BY_BACKEND[campaign.BACKEND_CPU],
                    {**spec().bench_params, "n_gen": 64})))

    def test_ranked_hard_cases_are_real_campaign_units(self):
        built = spec(blocks=2, ranked_units=ranked_units())
        self.assertEqual(built.ranked_unit_ids,
                         ("normal-prefill-512", "hard-prefill-511"))
        self.assertEqual(built.anti_short_circuit_units, ("hard-prefill-511",))
        self.assertEqual(built.ranked_unit_param_overrides["hard-prefill-511"]
                         ["n_prompt"], 511)
        self.assertEqual(set(built.physical_envelopes), set(built.ranked_unit_ids))
        self.assertEqual(len(built.to_dict()["ranked_units"]), 2)

    def test_ranked_manifest_requires_normal_and_hard_real_commands(self):
        with self.assertRaisesRegex(ValueError, "normal control"):
            spec(ranked_units=(ranked_units()[1],))
        relabelled = campaign.RankedUnitSpec(
            unit_id="hard-relabel", kind=campaign.RANKED_UNIT_ANTI_SHORT_CIRCUIT,
            params={"n_prompt": 512},
            physical_envelope=physical_envelope(shape_id="hard-relabel"))
        with self.assertRaisesRegex(ValueError, "same recipe params"):
            spec(blocks=2, ranked_units=(ranked_units()[0], relabelled))

    def test_ranked_manifest_parser_is_strict(self):
        payload = {"schema": campaign.RANKED_UNITS_SCHEMA,
                   "units": [unit.to_dict() for unit in ranked_units()]}
        parsed = campaign.ranked_units_from_mapping(payload)
        self.assertEqual(parsed, ranked_units())
        payload["surprise"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            campaign.ranked_units_from_mapping(payload)

    def test_a_claim_id_namespace_cannot_be_used_for_a_candidate(self):
        with self.assertRaises(ValueError):
            spec(candidate_id="akclaim-0001")

    def test_the_cpu_list_is_derived_from_the_argv_that_pins_it(self):
        """Never retyped: production drifted off interleave on a 1.7% warm A/B."""
        from .evaluator import recipes
        prefix = list(recipes.CANONICAL_PREFIX)
        built = spec()
        self.assertEqual(built.cpu_list, prefix[prefix.index("-c") + 1])
        self.assertIn(f"-c {built.cpu_list}",
                      " ".join(campaign.render_bench_commands(built)["candidate"]["argv"]))

    def test_the_gpu_cell_claims_the_cores_it_actually_pins(self):
        """THE BITE. The GPU cell pins 184-191, not the canonical 0-95.

        Claiming the canonical prefix here would leave every measured core
        unprotected while looking, in every journal field, exactly like a
        claimed run — and `MicrobenchRunner._attest_claim` would refuse it an
        hour into the claim window, after the build.
        """
        from .evaluator import recipes
        gpu = spec(backend="llama_gpu", devices=("ROCm0",),
                   device_names=("AMD Instinct MI210",))
        canonical = list(recipes.CANONICAL_PREFIX)
        self.assertNotEqual(gpu.cpu_list, canonical[canonical.index("-c") + 1])
        self.assertIn(f"-c {gpu.cpu_list}",
                      " ".join(campaign.render_bench_commands(gpu)["candidate"]["argv"]))

    def test_both_arms_pin_the_same_footprint(self):
        for built in (spec(), spec(backend="llama_gpu", devices=("ROCm0",),
                                   device_names=("AMD Instinct MI210",))):
            with self.subTest(backend=built.backend):
                rendered = campaign.render_bench_commands(built)
                self.assertEqual(rendered["anchor"]["cpu_list"],
                                 rendered["candidate"]["cpu_list"])

    def test_a_gpu_campaign_must_name_a_device(self):
        with self.assertRaises(ValueError):
            spec(backend="llama_gpu")

    def test_a_gpu_cell_is_not_satisfied_by_a_cpu_device_name(self):
        """THE BITE. 'Device 0: CPU' is what evaluator/devices.py exists to catch."""
        with self.assertRaises(ValueError) as raised:
            spec(backend="llama_gpu", devices=("ROCm0",),
                 device_names=("Device 0: CPU",))
        self.assertIn("GPU lane", str(raised.exception))

    def test_an_unrecognised_device_name_is_also_refused(self):
        """COULD_NOT_CHECK establishes neither lane, so it is not a pass either."""
        with self.assertRaises(ValueError):
            spec(backend="llama_gpu", devices=("ROCm0",), device_names=("thing-0",))

    def test_a_gpu_campaign_with_a_gpu_device_is_accepted(self):
        """Compliant-path control for the device-vocabulary guard."""
        built = spec(backend="llama_gpu", devices=("ROCm0",),
                     device_names=("AMD Instinct MI210",))
        self.assertEqual(built.devices, ("ROCm0",))
        self.assertEqual(built.bench_params["device_id"], "ROCm0")

    def test_a_scratch_journal_root_is_refused(self):
        """The 2026-07-04 win was written to /mnt/raid0/llm/tmp/ and is gone."""
        with self.assertRaises(Exception):
            spec(journal_root="/mnt/raid0/llm/tmp/ak-journal")

    def test_n_of_one_is_refused(self):
        with self.assertRaises(ValueError):
            spec(blocks=1)

    def test_the_drift_bound_follows_the_cell(self):
        decode = spec(recipe_id="t1b.llama_cpu.llama_bench_decode.v1")
        prefill = spec()
        self.assertNotEqual(decode.drift_bound, prefill.drift_bound)
        self.assertEqual(decode.drift_bound,
                         campaign.DRIFT_BOUND_BY_METRIC["decode_tokens_per_s"])

    def test_a_missing_required_recipe_parameter_is_refused_before_the_claim(self):
        with self.assertRaises(Exception):
            spec(model=None)


# =============================================================================
# 7b. The exit code, and the second Ctrl-C
# =============================================================================

class TestAnUnprovenProductionTreeIsNotASuccess(unittest.TestCase):
    """`_finish` turns a proof that RAISED into COULD_NOT_CHECK and says, in its
    own reason, that this "outranks everything else in this run" — and the run
    then exited 0 anyway, because `ok` only rejected an outright FAIL.

    COULD_NOT_CHECK is exactly what a proof reaches when the thing it inspects
    has been disturbed. Treating it as clean is the fail-open shape.
    """

    class Ops(SpyOps):
        def __init__(self, *, proof, **kw):
            super().__init__(**kw)
            self._proof = proof

        def prove_production_unchanged(self, spec_):
            self.calls.append("prove_production_unchanged")
            if isinstance(self._proof, BaseException):
                raise self._proof
            return self._proof

    def campaign_run(self, proof, **kw):
        ops = self.Ops(proof=proof, pairs=pairs_from_positions(
            TG128_OVER_TEN_POSITIONS, candidate_factor=1.08, orders=BALANCED), **kw)
        return campaign.run_campaign(spec(), ops)

    def test_a_proof_that_raised_is_not_a_clean_run(self):
        """THE BITE: this returned ok=True, and `main` exited 0."""
        result = self.campaign_run(RuntimeError("git rev-parse died"))
        self.assertEqual(result.production_unchanged.outcome, schemas.COULD_NOT_CHECK)
        self.assertFalse(result.ok)

    def test_an_unfingerprinted_tree_is_not_a_clean_run(self):
        result = self.campaign_run(schemas.Check(schemas.COULD_NOT_CHECK, ("no fingerprint",)))
        self.assertFalse(result.ok)

    def test_the_compliant_path_is_still_a_success(self):
        """CONTROL: a PASS on an executing run is ok, KEEP or REVERT alike."""
        result = self.campaign_run(schemas.Check(schemas.PASS, ("byte-identical",)))
        self.assertTrue(result.ok)
        self.assertTrue(result.executed)

    def test_a_dry_run_is_exempt_because_it_fingerprinted_nothing(self):
        """CONTROL, and the one that stops this becoming 'always FAIL': a
        composition pass reads nothing from the frozen trees and claims
        nothing about them, so its COULD_NOT_CHECK is honest and clean."""
        out = io.StringIO()
        result = campaign.run_campaign(spec(), campaign.DryRunOps(out=out))
        self.assertFalse(result.executed)
        self.assertEqual(result.production_unchanged.outcome, schemas.COULD_NOT_CHECK)
        self.assertTrue(result.ok)

    def test_the_record_says_which_kind_of_run_it_was(self):
        self.assertTrue(self.campaign_run(schemas.Check(schemas.PASS, ()))
                        .to_dict()["executed"])


class TestAResultThatCouldNotBeWrittenDownIsNotASuccess(unittest.TestCase):
    """A journal failure printed a warning to stderr and the process exited 0.

    A wrapper reading the status learned nothing, which is the shape of the
    incident the journal exists for: AutoPilot lost 232 trials / ~16 days to a
    loop that held its results in memory.
    """

    ROOT = "/mnt/raid0/llm/epyc-inference-research/data/ak-test-journal"

    def _run(self, *, journal_root, raises):
        ops = SpyOps(pairs=pairs_from_positions(TG128_OVER_TEN_POSITIONS,
                                                candidate_factor=1.08, orders=BALANCED))
        if raises:
            def explode(spec_, payload):
                ops.calls.append("journal")
                raise OSError("read-only file system")
            ops.journal = explode
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            return campaign.run_campaign(spec(journal_root=journal_root), ops)

    def test_a_journal_that_failed_makes_the_campaign_not_ok(self):
        """THE BITE."""
        result = self._run(journal_root=self.ROOT, raises=True)
        self.assertIsNotNone(result.journal_error)
        self.assertFalse(result.ok)
        self.assertEqual(result.to_dict()["journal_error"][:7], "OSError")

    def test_the_result_is_still_returned(self):
        """The failure must not HIDE the result — only the exit code changes."""
        result = self._run(journal_root=self.ROOT, raises=True)
        self.assertEqual(result.state, campaign.STATE_DECIDED)
        self.assertIsNotNone(result.decision)

    def test_a_campaign_that_asked_for_no_journal_is_unaffected(self):
        """CONTROL: `--journal-root` is optional and its absence is not a failure."""
        result = self._run(journal_root=None, raises=True)
        self.assertIsNotNone(result.journal_error)
        self.assertTrue(result.ok)

    def test_the_compliant_path_journals_and_is_ok(self):
        """CONTROL."""
        result = self._run(journal_root=self.ROOT, raises=False)
        self.assertIsNone(result.journal_error)
        self.assertTrue(result.ok)


class TestProposalIsDurableBeforeHostWork(unittest.TestCase):
    def setUp(self):
        data_root = Path(__file__).resolve().parents[3] / "data"
        self.tempdir = tempfile.TemporaryDirectory(prefix="ak-proposal-test-", dir=data_root)
        self.addCleanup(self.tempdir.cleanup)

    def test_identical_resume_reuses_one_fsynced_proposal_event(self):
        built = spec(journal_root=self.tempdir.name, proposal=proposal_manifest())
        ops = campaign.HostOps()
        first = ops.record_proposal(built)
        second = ops.record_proposal(built)
        self.assertEqual(first, second)
        book = campaign.journal_module.Journal(
            self.tempdir.name, campaign_id=built.campaign_id
        )
        proposals = [
            entry
            for entry in book.read_all()
            if entry.kind == campaign.journal_module.KIND_PROPOSAL_RECORDED
        ]
        self.assertEqual(len(proposals), 1)

    def test_same_proposal_id_with_different_bytes_is_refused(self):
        first = spec(journal_root=self.tempdir.name, proposal=proposal_manifest())
        ops = campaign.HostOps()
        ops.record_proposal(first)
        changed = proposal_manifest()
        changed["hypothesis"] = "different hypothesis under the same id"
        second = spec(journal_root=self.tempdir.name, proposal=changed)
        with self.assertRaisesRegex(RuntimeError, "different bytes"):
            ops.record_proposal(second)


class TestASecondInterruptDoesNotStrandTheRest(unittest.TestCase):
    """`run_campaign` catches `BaseException` because Ctrl-C is the realistic
    early exit. The releases themselves caught only `Exception`, so the SECOND
    Ctrl-C — the one during teardown — stranded every release after the one it
    landed on, the claim among them.
    """

    def test_an_interrupt_in_one_release_does_not_strand_the_others(self):
        """THE BITE."""
        ledger = campaign.ResourceLedger()
        released = []
        ledger.hold("claim", lambda: released.append("claim"))
        ledger.hold("worktree", _interrupting)
        records = ledger.release_all()
        self.assertEqual(released, ["claim"])
        self.assertEqual([r.name for r in records], ["worktree", "claim"])
        self.assertEqual([r.released for r in records], [False, True])

    def test_an_interrupt_in_keep_or_revert_still_releases_the_claim(self):
        """THE BITE, one layer up: `keep_or_revert` runs BEFORE the releases."""
        ops = SpyOps(pairs=pairs_from_positions(TG128_OVER_TEN_POSITIONS,
                                                candidate_factor=1.08, orders=BALANCED))
        ops.keep_or_revert = _interrupting
        result = campaign.run_campaign(spec(), ops)
        self.assertIn("release_claim", ops.calls)
        self.assertTrue(all(r.released for r in result.releases))

    def test_a_clean_release_is_still_recorded_as_one(self):
        """CONTROL."""
        ledger = campaign.ResourceLedger()
        ledger.hold("claim", lambda: "released")
        records = ledger.release_all()
        self.assertEqual([r.released for r in records], [True])


def _interrupting(*_args, **_kwargs):
    raise KeyboardInterrupt()


class TestExecuteRefusesAnOpsThatCannotFinishARun(unittest.TestCase):
    """Source campaigns have proposal-specific seams; IQK parameter replay does not.

    Each missing source seam would otherwise raise where it is reached - after the region
    claim, after the worktree, after a forty-minute build - so the cost of
    discovering it was the claim window. It is knowable at argv time.
    """

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        manifest = Path(self.tempdir.name) / "proposal.json"
        manifest.write_text(json.dumps(proposal_manifest()), encoding="utf-8")
        envelope = Path(self.tempdir.name) / "physical-envelope.json"
        envelope.write_text(json.dumps(physical_envelope().to_dict()), encoding="utf-8")
        ranked = Path(self.tempdir.name) / "ranked-units.json"
        ranked.write_text(json.dumps({
            "schema": campaign.RANKED_UNITS_SCHEMA,
            "units": [unit.to_dict() for unit in ranked_units()],
        }), encoding="utf-8")
        self.ranked_manifest = ranked
        calibration = write_calibration_bundle(
            Path(self.tempdir.name) / "calibration")
        self.argv = [
            "--model", MODEL, "--campaign-id", "ak-test", "--candidate-id", "akc-test",
            "--execute", "--i-hold-the-host",
            "--proposal-manifest", str(manifest),
            "--calibration-bundle", str(calibration),
            "--physical-envelope", str(envelope),
        ]

    def test_stock_source_host_ops_requires_mutator_and_operator_frequency(self):
        """THE BITE."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = campaign.main(self.argv, out=io.StringIO(),
                                 ops=campaign.HostOps())
        self.assertEqual(code, 2)
        for seam in ("apply_candidate", "t0_evidence", "nominal_khz"):
            self.assertIn(seam, err.getvalue())
        for closed in ("_anchor_identity_for_bench",):
            self.assertNotIn(f"  {closed}:", err.getvalue())

    def test_an_override_clears_its_own_seam(self):
        """Derived from what is bound, not from a flag someone has to flip."""
        class Wired(campaign.HostOps):
            def apply_candidate(self, spec_, tree):
                return None

            def _anchor_identity_for_bench(self, spec_):
                return None

        self.assertEqual(Wired(t0_evidence=lambda **kw: {}, nominal_khz=2_900_000)
                         .unimplemented_seams(), ())
        self.assertIn("apply_candidate", campaign.HostOps().unimplemented_seams())

    def test_parameter_proposal_has_a_no_source_mutation_receipt(self):
        class Tree:
            def unified_diff_from_source(self):
                return ""

        built = spec(proposal=iqk_parameter_proposal())
        receipt = campaign.HostOps(nominal_khz=2_900_000).apply_candidate(built, Tree())
        self.assertFalse(receipt["source_mutated"])
        self.assertEqual(receipt["candidate"], {"ggml_iqk": "1"})
        self.assertEqual(receipt["anchor"], {"ggml_iqk": "0"})

    def test_parameter_proposal_closes_all_but_the_operator_frequency(self):
        built = spec(proposal=iqk_parameter_proposal())
        self.assertEqual(campaign.HostOps().unimplemented_seams(built),
                         ("nominal_khz",))

    def test_host_paired_blocks_calls_the_plan_schedule_method_before_spending(self):
        """A plan derives its schedule; it does not expose it as a field."""
        built = spec(proposal=iqk_parameter_proposal())
        ops = campaign.HostOps(nominal_khz=2_900_000)
        ops._claim_binding = mock.Mock(microbench_claim=object())
        ops._build_state = {"tree": mock.Mock(path=mock.Mock(path=self.tempdir.name))}
        ops._spawner = object()
        command = mock.Mock(binding=object(), receipt=object())
        anchor = mock.Mock(tool="llama-bench")
        schedule = mock.Mock()
        schedule.orders.return_value = ("anchor_first", "candidate_first")
        plan = mock.Mock(campaign_seed="schedule-method-regression")
        plan.schedule.return_value = schedule
        run = object()
        with mock.patch.object(ops, "_construct", side_effect=(command, command)), \
                mock.patch.object(ops, "_anchor_identity_for_bench", return_value=anchor), \
                mock.patch.object(ops, "_t1_evaluation_request"), \
                mock.patch.object(ops, "_candidate_sandbox_policy"), \
                mock.patch.object(ops, "_completed_run_ledger"), \
                mock.patch.object(campaign.microbench, "MicrobenchPlan", return_value=plan), \
                mock.patch.object(campaign.microbench, "MicrobenchRunner") as runner_type, \
                mock.patch.object(campaign, "pairs_from_run", return_value=()) as pairs:
            runner_type.return_value.run.return_value = run
            self.assertEqual(ops.run_paired_blocks(built, object(), object()), ())
        plan.schedule.assert_called_once_with()
        schedule.orders.assert_called_once_with(built.blocks)
        pairs.assert_called_once_with(run)

    def test_parameter_proposal_rejects_source_prerequisite_package(self):
        package = mock.Mock(
            spec=source_prerequisite_package.SourcePrerequisitePackage)
        with self.assertRaisesRegex(ValueError, "parameter campaigns"):
            spec(proposal=iqk_parameter_proposal(),
                 source_prerequisite_package=package)

    def test_host_build_makes_compiler_warnings_fatal(self):
        source = campaign.worktree.SandboxPath.create(
            str(Path(self.tempdir.name) / "source"), production_trees=())
        build_dir = campaign.worktree.SandboxPath.create(
            str(Path(self.tempdir.name) / "build"), production_trees=())
        tree = mock.Mock(path=source)
        result = object()
        with mock.patch.object(
                campaign.worktree, "default_build_dir", return_value=build_dir), \
                mock.patch.object(
                    campaign.worktree, "run_build", return_value=result) as run:
            self.assertIs(
                campaign.HostOps(nominal_khz=2_900_000).build(spec(), tree), result)
        plan = run.call_args.args[0]
        self.assertEqual(dict(plan.effective_defines)["LLAMA_FATAL_WARNINGS"], "ON")
        self.assertEqual(dict(plan.effective_defines)["LLAMA_BUILD_EXAMPLES"], "ON")
        self.assertIn(campaign.T0_GENERATION_TOOL, plan.targets)

    def test_failed_build_is_refused_before_artifact_hashing(self):
        """THE BITE: a failed compiler result cannot enter T0 evidence assembly."""
        failed = mock.Mock(spec=campaign.worktree.BuildResult)
        failed.succeeded = False
        failed.exit_code = 2
        ops = campaign.HostOps(nominal_khz=2_900_000)
        ops._claim_binding = object()
        ops._build_state = {"result": failed, "tree": object(), "plan": object()}

        with mock.patch.object(
                campaign, "_source_tree_digest",
                side_effect=AssertionError("artifact hashing must not start")):
            with self.assertRaisesRegex(RuntimeError,
                                        "failed build before artifact hashing.*exit_code=2"):
                ops.run_t0(spec(), failed)

    def test_successful_build_without_t0_artifacts_is_refused_before_hashing(self):
        """THE BITE: a zero exit alone cannot license absent CMake outputs."""
        succeeded = mock.Mock(spec=campaign.worktree.BuildResult)
        succeeded.succeeded = True
        plan = mock.Mock()
        plan.build_dir.path = str(Path(self.tempdir.name) / "empty-build")
        ops = campaign.HostOps(nominal_khz=2_900_000)
        ops._claim_binding = object()
        ops._build_state = {"result": succeeded, "tree": object(), "plan": plan}

        with mock.patch.object(
                campaign, "_source_tree_digest",
                side_effect=AssertionError("artifact hashing must not start")):
            with self.assertRaisesRegex(
                    RuntimeError,
                    "missing/unusable required build artifacts before artifact hashing.*llama-completion"):
                ops.run_t0(spec(), succeeded)

    def test_executed_t0_constructs_the_complete_cpu_evaluator_policy(self):
        """The live T0 path must not instantiate the no-default policy bare.

        T0Policy intentionally rejects an implicit policy.  Pin the campaign's
        protocol choices here so a constructor-field addition fails this test
        before a live campaign can reach T0.
        """
        policy = campaign.HostOps._t0_evaluator_policy(spec())
        self.assertEqual(policy.required_backend_ops,
                         campaign.correctness.MANDATORY_BACKEND_OPS)
        self.assertEqual(policy.diff_ceiling.backend, campaign.BACKEND_CPU)
        self.assertEqual(policy.determinism_min_runs, 2)
        self.assertEqual(policy.policy_ref, "ak-policy/v1")

    def test_t0_generation_plan_binds_the_campaign_model(self):
        """T0 binds the direct no-socket generator to the measured model."""
        built = spec()
        plan = campaign.HostOps._t0_generation_plan(built)
        self.assertEqual(plan.extra_argv, ("-m", built.model))
        invocation = campaign.t0_provider.build_generation_invocation(
            binary="/tmp/llama-completion", library_path="/tmp", plan=plan, base_env=())
        self.assertEqual(invocation.argv[invocation.argv.index("-m") + 1], built.model)

    def test_t0_capture_sink_is_durable_and_outside_candidate_sandbox(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(campaign.storage, "assert_not_scratch", return_value=root):
                built = spec(journal_root=root)
                sink = campaign.HostOps._t0_capture_sink(built)
            self.assertEqual(Path(sink._root), Path(root, "t0-captures"))
            self.assertNotIn("candidate-sandbox", str(sink._root))

    def test_parameter_t0_adapter_derives_the_nonbehavioural_gate_surfaces(self):
        built = spec(proposal=iqk_parameter_proposal())
        tree = mock.Mock()
        tree.path.path = str(Path(self.tempdir.name) / "candidate")
        # Production T0 builds run from a committed, detached snapshot rather
        # than claiming the source actor's branch.  A branch label here would
        # hide the exact runtime condition that the worktree API represents as
        # ``None``.
        tree.branch = None
        tree.head_commit.return_value = "b" * 40
        tree.unified_diff_from_source.return_value = ""
        plan = mock.Mock()
        plan.build_dir.path = str(Path(self.tempdir.name) / "build")
        ops = campaign.HostOps(nominal_khz=2_900_000)
        ops._build_state = {"tree": tree, "plan": plan}
        anchor_capture = campaign.t0_provider.AnchorCapture(
            source_commit=campaign.MEASUREMENT_COMMIT,
            binary_sha256=schemas.content_hash({"tool": campaign.T0_GENERATION_TOOL}),
            linkage_sha256=schemas.content_hash({"libs": "anchor"}))
        ops._t0_anchor_binding = campaign.chain.bind_anchor(
            anchor_capture, tool=campaign.T0_GENERATION_TOOL)
        library_capture = campaign.t0_provider.AnchorCapture(
            source_commit=campaign.MEASUREMENT_COMMIT,
            binary_sha256=schemas.content_hash({"tool": "libggml.so.0"}),
            linkage_sha256=anchor_capture.linkage_sha256)
        symbols = object()
        diff = object()
        projected = {
            "symbols": "projected-symbols",
            "diff": "projected-diff",
            "change_surface": "projected-surface",
            "projection_checks": (),
        }
        with mock.patch.object(
                campaign.t0_provider, "capture_anchor_identity",
                return_value=library_capture) as capture_identity, \
                mock.patch.object(
                    ops, "_construct", return_value=mock.Mock(env={
                        "LD_LIBRARY_PATH": campaign.MEASUREMENT_BUILD_ROOT + "/bin"})), \
                mock.patch.object(
                    campaign.chain, "iqk_parameter_symbol_evidence",
                    return_value=symbols) as symbol_adapter, \
                mock.patch.object(
                    campaign.chain, "diff_policy_evidence",
                    return_value=diff) as diff_policy_evidence, \
                mock.patch.object(
                    campaign.chain, "t0_plan_evidence",
                    return_value=projected) as project:
            evidence = ops._parameter_t0_evidence(
                built, identity=object(), build_evidence=object())
        self.assertEqual(evidence["symbols"], "projected-symbols")
        self.assertNotIn("reference", evidence)
        self.assertEqual(
            capture_identity.call_args.kwargs["parameter_env"],
            (("GGML_IQK", "0"),))
        self.assertEqual(
            symbol_adapter.call_args.kwargs["candidate_root"], tree.path.path)
        self.assertEqual(
            diff_policy_evidence.call_args.kwargs["branch_name"], "detached@" + "b" * 40)
        surface = project.call_args.kwargs["change_surface"].surface
        self.assertTrue(surface.derived_touches_dispatch)
        self.assertFalse(surface.derived_touches_memory)
        self.assertFalse(surface.derived_touches_threading)
        self.assertFalse(surface.derived_touches_persistent_state)
        self.assertEqual(surface.derived_ops, built.t0_ops)

    def test_parameter_t0_adapter_records_detached_snapshot_head_not_a_branch(self):
        """T0 must support the clean detached snapshot it builds from.

        A parameter arm has no source patch, but still builds from an immutable
        snapshot worktree.  Such worktrees deliberately have ``branch is None``;
        treating them as branch worktrees caused r16 to crash before T0.  The
        provenance label must name the snapshot HEAD rather than invent a branch.
        """
        built = spec(proposal=iqk_parameter_proposal())
        tree = mock.Mock()
        tree.path.path = str(Path(self.tempdir.name) / "candidate")
        tree.branch = None
        tree.head_commit.return_value = "a" * 40
        tree.unified_diff_from_source.return_value = ""
        plan = mock.Mock()
        plan.build_dir.path = str(Path(self.tempdir.name) / "build")
        ops = campaign.HostOps(nominal_khz=2_900_000)
        ops._build_state = {"tree": tree, "plan": plan}
        anchor_capture = campaign.t0_provider.AnchorCapture(
            source_commit=campaign.MEASUREMENT_COMMIT,
            binary_sha256=schemas.content_hash({"tool": "llama-cli"}),
            linkage_sha256=schemas.content_hash({"libs": "anchor"}))
        ops._t0_anchor_binding = campaign.chain.bind_anchor(
            anchor_capture, tool="llama-cli")
        library_capture = campaign.t0_provider.AnchorCapture(
            source_commit=campaign.MEASUREMENT_COMMIT,
            binary_sha256=schemas.content_hash({"tool": "libggml.so.0"}),
            linkage_sha256=anchor_capture.linkage_sha256)
        projected = {"symbols": "projected-symbols", "diff": "projected-diff",
                     "change_surface": "projected-surface", "projection_checks": ()}
        with mock.patch.object(campaign.t0_provider, "capture_anchor_identity",
                               return_value=library_capture), \
                mock.patch.object(ops, "_construct", return_value=mock.Mock(env={
                    "LD_LIBRARY_PATH": campaign.MEASUREMENT_BUILD_ROOT + "/bin"})), \
                mock.patch.object(campaign.chain, "iqk_parameter_symbol_evidence"), \
                mock.patch.object(campaign.chain, "diff_policy_evidence",
                                  return_value=mock.Mock()) as diff_policy_evidence, \
                mock.patch.object(campaign.chain, "t0_plan_evidence",
                                  return_value=projected):
            ops._parameter_t0_evidence(built, identity=object(), build_evidence=object())
        self.assertEqual(diff_policy_evidence.call_args.kwargs["branch_name"],
                         "detached@" + "a" * 40)
        tree.head_commit.assert_called_once_with()

    def test_parameter_proposal_refuses_a_source_diff(self):
        class Tree:
            def unified_diff_from_source(self):
                return "diff --git a/a b/a\n+changed\n"

        with self.assertRaisesRegex(RuntimeError, "no longer one-factor"):
            campaign.HostOps(nominal_khz=2_900_000).apply_candidate(
                spec(proposal=iqk_parameter_proposal()), Tree())

    def test_source_proposal_still_requires_a_campaign_specific_mutator(self):
        missing = campaign.HostOps(nominal_khz=2_900_000).unimplemented_seams(
            spec(proposal=proposal_manifest()))
        self.assertIn("apply_candidate", missing)
        self.assertIn("source_prerequisites", missing)

    def test_source_proposal_archive_resume_reaches_t0_pass(self):
        """A real CampaignSpec/build identity can satisfy, not only refuse, the seam."""
        import sys
        kernel_rnd = str(Path(__file__).resolve().parent.parent)
        if kernel_rnd not in sys.path:
            sys.path.insert(0, kernel_rnd)
        from autokernel import campaign as live_campaign
        from autokernel import source_prerequisite_package as live_package
        from autokernel.evaluator.test_correctness import (
            evidence as t0_evidence, policy as t0_policy, request as t0_request)
        from .test_source_prerequisite_package import package as prerequisite_package

        request = t0_request()
        package_mapping = prerequisite_package()
        package_mapping["candidate_source_sha256"] = request.artifact.source_sha256
        package_mapping["candidate_binary_sha256"] = request.artifact.binary_sha256
        package_mapping["evaluator_bundle_sha256"] = request.evaluator.bundle_sha256
        package_mapping["package_sha256"] = live_package.package_sha256(package_mapping)
        archived = live_package.SourcePrerequisitePackage.from_mapping(package_mapping)
        built_spec = live_campaign.CampaignSpec(
            campaign_id="ak-test", candidate_id="akc-test",
            candidate_ref="candidate.patch", model=MODEL,
            proposal=proposal_manifest(), source_prerequisite_package=archived)
        identity = mock.Mock(spec=live_campaign.worktree.BuildIdentity)
        identity.snapshot_sha256 = request.artifact.source_sha256
        candidate = mock.Mock(spec=live_campaign.t0_provider.CandidateBuild)
        candidate.test_backend_ops = str(Path(self.tempdir.name) / "test-backend-ops")
        evaluator = request.evaluator

        with mock.patch.object(
                live_campaign.storage, "hash_file",
                return_value=request.artifact.binary_sha256):
            bound = live_campaign.HostOps()._source_prerequisites_for_t0(
                built_spec, identity=identity, candidate=candidate, evaluator=evaluator)
        report = live_campaign.correctness.evaluate_t0(
            request,
            t0_evidence(source_candidate=True, source_prerequisites=bound),
            t0_policy())
        self.assertEqual(report.failed, ())
        self.assertEqual(report.unevaluated, ())

    def test_a_runnable_ops_is_not_refused(self):
        """CONTROL: the guard must not forbid its own compliant path."""
        positions = drifting(52.76, 50.52, 24)
        orders = ("anchor_first", "candidate_first") * 6
        ops = SpyOps(pairs=pairs_from_positions(
            positions, candidate_factor=1.08, orders=orders))
        code = campaign.main(self.argv, out=io.StringIO(), ops=ops)
        self.assertEqual(code, 0)
        self.assertIn("run_paired_blocks", ops.calls)

    def test_a_dry_run_is_never_refused_for_an_unwired_seam(self):
        """CONTROL: composing the loop needs none of them."""
        out = io.StringIO()
        self.assertEqual(campaign.main(["--model", MODEL], out=out,
                                       ops=campaign.DryRunOps(out=out)), 0)

    def test_execute_without_a_proposal_is_refused_before_ops(self):
        ops = SpyOps()
        code = campaign.main(
            ["--model", MODEL, "--execute", "--i-hold-the-host"],
            out=io.StringIO(),
            ops=ops,
        )
        self.assertEqual(code, 2)
        self.assertEqual(ops.calls, [])

    def test_iqk_execute_without_capture_plan_is_refused_before_ops(self):
        proposal_path = Path(self.tempdir.name) / "iqk-proposal.json"
        proposal_path.write_text(
            json.dumps(iqk_parameter_proposal()), encoding="utf-8")
        argv = list(self.argv)
        index = argv.index("--proposal-manifest")
        argv[index + 1] = str(proposal_path)
        ops = SpyOps()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = campaign.main(argv, out=io.StringIO(), ops=ops)
        self.assertEqual(code, 2)
        self.assertEqual(ops.calls, [])
        self.assertIn("--least-commitment-capture-plan", err.getvalue())

    def test_malformed_prerequisite_package_is_refused_before_ops(self):
        path = Path(self.tempdir.name) / "bad-prerequisites.json"
        path.write_text("not json", encoding="utf-8")
        argv = [*self.argv, "--source-prerequisite-package", str(path)]
        ops = SpyOps()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = campaign.main(argv, out=io.StringIO(), ops=ops)
        self.assertEqual(code, 2)
        self.assertEqual(ops.calls, [])
        self.assertIn("--source-prerequisite-package", err.getvalue())

    def test_dry_run_prerequisite_archive_cannot_enter_execute_mode(self):
        from .test_source_prerequisite_package import package as package_fixture
        payload = package_fixture(mode="dry_run")
        path = Path(self.tempdir.name) / "dry-prerequisites.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        argv = [*self.argv, "--source-prerequisite-package", str(path)]
        ops = SpyOps()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = campaign.main(argv, out=io.StringIO(), ops=ops)
        self.assertEqual(code, 2)
        self.assertEqual(ops.calls, [])
        self.assertIn("dry_run mode", err.getvalue())

    def test_execute_without_a_physical_envelope_is_refused_before_ops(self):
        ops = SpyOps()
        argv = list(self.argv)
        index = argv.index("--physical-envelope")
        del argv[index:index + 2]
        code = campaign.main(argv, out=io.StringIO(), ops=ops)
        self.assertEqual(code, 2)
        self.assertEqual(ops.calls, [])

    def test_execute_accepts_ranked_units_instead_of_single_envelope(self):
        positions = drifting(52.76, 50.52, 24)
        orders = ("anchor_first", "candidate_first") * 6
        ops = SpyOps(pairs=pairs_from_positions(
            positions, candidate_factor=1.08, orders=orders))
        argv = list(self.argv)
        index = argv.index("--physical-envelope")
        argv[index:index + 2] = ["--ranked-units", str(self.ranked_manifest)]
        code = campaign.main(argv, out=io.StringIO(), ops=ops)
        self.assertEqual(code, 0)
        self.assertIn("run_paired_blocks", ops.calls)


# =============================================================================
# 8a. HostOps holds nothing it cannot release
#
# `HostOps` is the only class here that touches the host, and no line of it has
# ever been run. These tests substitute the module-level functions it calls —
# nothing is spawned, no lock is taken, no git command runs — and pin the one
# property whose absence is a leaked claim on a shared machine.
# =============================================================================

class FakeClaim:
    def __init__(self, name="region"):
        self.name = name
        self.released = 0

    def release(self):
        self.released += 1
        return self

    def receipt(self):
        return self

    def verify_held(self):
        return schemas.Check(schemas.PASS, ())

    def to_dict(self):
        return {"claim": self.name, "claim_id": self.name,
                "device_id": self.name, "released": self.released}


class TestTheClaimIsNeverHeldWithoutAReleaser(unittest.TestCase):
    """The ledger registers the releaser only once `acquire_claim` RETURNS.

    Everything after `acquire_cpu_region_claim` — the seam check, and every
    device claim after the first — can raise, and until this was fixed the
    region claim was then held by a process on its way out with no releaser
    anywhere. "Released by the ledger" was true only on the happy path.
    """

    def setUp(self):
        self.ops = campaign.HostOps()
        self.spec = spec()
        self.gpu_spec = spec(backend="llama_gpu", devices=("ROCm0", "ROCm1"),
                             device_names=("AMD Instinct MI210", "AMD Instinct MI210"))
        self.region = FakeClaim()
        patches = [
            mock.patch.object(campaign.cpu_region_claim, "RegionClaimJournal",
                              lambda *a, **k: object()),
            mock.patch.object(campaign.cpu_region_claim, "acquire_cpu_region_claim",
                              lambda *a, **k: self.region),
            mock.patch.object(campaign.chain, "bind_claim", lambda *a, **k: object()),
            mock.patch.object(
                campaign.device_claim, "check_device_claim_held",
                lambda *a, **k: schemas.Check(schemas.PASS, ())),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _seams(self, outcome):
        return mock.patch.object(
            campaign.chain, "check_claim_satisfies_both_seams",
            lambda *a, **k: schemas.Check(outcome, ("fake",)))

    def test_a_claim_that_fails_the_seam_check_is_released_here(self):
        """THE BITE. The seam check raises AFTER the flock is held."""
        with self._seams(schemas.FAIL):
            with self.assertRaises(RuntimeError):
                self.ops.acquire_claim(self.spec)
        self.assertEqual(self.region.released, 1,
                         "the region claim was acquired and never released")

    def test_the_compliant_path_holds_the_claim_it_returns(self):
        """CONTROL: a passing seam check must NOT release what it just took."""
        with self._seams(schemas.PASS):
            claim = self.ops.acquire_claim(self.spec)
        self.assertIs(claim, self.region)
        self.assertEqual(self.region.released, 0)

    def test_a_claim_that_cannot_be_reverified_is_released_before_return(self):
        self.region.verify_held = lambda: schemas.Check(
            schemas.FAIL, ("claim lock inode moved",))
        with self._seams(schemas.PASS):
            with self.assertRaisesRegex(RuntimeError, "inode moved"):
                self.ops.acquire_claim(self.spec)
        self.assertEqual(self.region.released, 1)

    def test_missing_stable_holder_fields_cannot_compare_as_same_holder(self):
        incomplete = {
            "region": {"claim_id": "claim-1", "holder_pid": None,
                       "holder_start_ticks": None, "holder_boot_id": None},
            "devices": []}
        self.assertEqual(campaign.HostOps._claim_holder_identity(incomplete), ())
        ops = campaign.HostOps()
        ops._claim_open_receipt = incomplete
        closed = json.loads(json.dumps(incomplete))
        check = ops._check_same_claim_holder(incomplete, closed)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("incomplete", " ".join(check.reasons))

    def test_a_device_claim_that_fails_releases_the_region_and_its_predecessor(self):
        """THE BITE, GPU: two devices, the second raises."""
        taken = []

        def acquire(device_id, **kwargs):
            if len(taken) == 1:
                raise RuntimeError("someone else holds ROCm1")
            claim = FakeClaim(device_id)
            taken.append(claim)
            return claim

        with self._seams(schemas.PASS), \
                mock.patch.object(campaign.device_claim, "acquire_device_claim", acquire):
            with self.assertRaises(RuntimeError):
                self.ops.acquire_claim(self.gpu_spec)
        self.assertEqual([c.released for c in taken], [1])
        self.assertEqual(self.region.released, 1)

    def test_device_claims_are_released_with_the_region_claim(self):
        """THE OTHER BITE: they were acquired and NEVER released at all.

        Nothing held a reference to them, so the flocks survived until the
        process died and the lock files went on naming a holder that no longer
        existed — the next GPU campaign meets a stale-grace wait or
        `DeviceClaimInconsistent`. A corpse holding the MI210.
        """
        taken = []

        def acquire(device_id, **kwargs):
            claim = FakeClaim(device_id)
            taken.append(claim)
            return claim

        with self._seams(schemas.PASS), \
                mock.patch.object(campaign.device_claim, "acquire_device_claim", acquire):
            claim = self.ops.acquire_claim(self.gpu_spec)
            self.assertEqual([c.released for c in taken], [0, 0])
            self.ops.release_claim(claim)
        self.assertEqual([c.released for c in taken], [1, 1])
        self.assertEqual(self.region.released, 1)

    def test_a_device_release_that_raises_does_not_strand_the_region_claim(self):
        class Stubborn(FakeClaim):
            def release(self):
                raise RuntimeError("the device flock could not be released")

        stubborn = Stubborn("ROCm0")
        with self._seams(schemas.PASS), \
                mock.patch.object(campaign.device_claim, "acquire_device_claim",
                                  lambda *a, **k: stubborn):
            claim = self.ops.acquire_claim(spec(backend="llama_gpu", devices=("ROCm0",),
                                                device_names=("AMD Instinct MI210",)))
            record = self.ops.release_claim(claim)
        self.assertEqual(self.region.released, 1)
        self.assertIn("error", record["devices"][0])

    def test_a_cpu_campaign_claims_no_device(self):
        """CONTROL: the device loop must not fire on the CPU cell."""
        def refuse(*_a, **_k):
            raise AssertionError("a llama_cpu campaign claimed a device")

        with self._seams(schemas.PASS), \
                mock.patch.object(campaign.device_claim, "acquire_device_claim", refuse):
            self.ops.acquire_claim(self.spec)

    def test_the_device_claim_declares_its_hold_window(self):
        """THE BITE. Without `max_hold_s` the device claim writes no `expires_at`.

        `check_claim_expiry()` then returns COULD_NOT_CHECK forever instead of
        FAIL — the one expiry check this fleet already owns, disarmed for exactly
        the claim that monopolised the MI210 on the night of 2026-08-11/12. The
        CPU claim three lines above had been declaring its window all along.

        The value is asserted to be the SAME as the region claim's: both are taken
        in one transaction, for one campaign, and released together, so two
        different deadlines would be a defect by construction.
        """
        seen = []

        def acquire(device_id, **kwargs):
            seen.append(kwargs)
            return FakeClaim(device_id)

        with self._seams(schemas.PASS), \
                mock.patch.object(campaign.device_claim, "acquire_device_claim", acquire):
            self.ops.acquire_claim(self.gpu_spec)

        self.assertEqual(len(seen), 2, "both devices should have been claimed")
        for kwargs in seen:
            self.assertIn("max_hold_s", kwargs,
                          "the device claim declares no maximum hold, so expiry can "
                          "never be evaluated on it")
            self.assertEqual(kwargs["max_hold_s"], float(self.gpu_spec.max_hold_s))

    def test_a_raised_hold_window_moves_both_claims_together(self):
        """CONTROL: the window is a SPEC value, not a constant baked in beside it.

        A guard that only checked "some number was passed" would pass against a
        hard-coded literal, and the CPU and device claims would then drift apart
        the first time a campaign legitimately needs longer.
        """
        seen, region_kwargs = [], {}

        def acquire_device(device_id, **kwargs):
            seen.append(kwargs)
            return FakeClaim(device_id)

        def acquire_region(*_a, **kwargs):
            region_kwargs.update(kwargs)
            return self.region

        longer = spec(backend="llama_gpu", devices=("ROCm0",),
                      device_names=("AMD Instinct MI210",), max_hold_s=9 * 3600)
        with self._seams(schemas.PASS), \
                mock.patch.object(campaign.device_claim, "acquire_device_claim",
                                  acquire_device), \
                mock.patch.object(campaign.cpu_region_claim, "acquire_cpu_region_claim",
                                  acquire_region):
            self.ops.acquire_claim(longer)

        self.assertEqual(seen[0]["max_hold_s"], float(9 * 3600))
        self.assertEqual(region_kwargs["max_hold_s"], seen[0]["max_hold_s"],
                         "the two claims covering one window declared different deadlines")


class TestEveryDeviceClaimSiteDeclaresItsWindow(unittest.TestCase):
    """Proved from `campaign.py`'s AST, not promised in a comment.

    Same one-door discipline as `audit_falsifier_required_before_claim`: a second
    acquisition growing beside the first must not be able to reintroduce the
    undeclared-window defect just by not passing the keyword.
    """

    SOURCE = Path(campaign.__file__)

    def _call_sites(self, source: str) -> list:
        tree = ast.parse(source)
        sites = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name == "acquire_device_claim":
                sites.append(node)
        return sites

    def test_every_call_site_passes_max_hold_s(self):
        sites = self._call_sites(self.SOURCE.read_text(encoding="utf-8"))
        self.assertTrue(sites, "no acquire_device_claim call site found; the guard "
                               "would pass vacuously")
        for node in sites:
            kwargs = {kw.arg for kw in node.keywords}
            self.assertIn("max_hold_s", kwargs,
                          f"acquire_device_claim at line {node.lineno} declares no "
                          f"max_hold_s, so its claim can never be checked for expiry")

    def test_the_guard_fails_on_a_call_site_that_omits_it(self):
        """CONTROL: the guard must actually bite, not merely find nothing to check."""
        doctored = (
            "def f(journal):\n"
            "    return device_claim.acquire_device_claim('mi210_0', purpose='p',\n"
            "                                            campaign_id='c', journal=journal)\n"
        )
        sites = self._call_sites(doctored)
        self.assertEqual(len(sites), 1)
        self.assertNotIn("max_hold_s", {kw.arg for kw in sites[0].keywords})


class TestTheWorktreeIsNotLeftBehind(unittest.TestCase):
    """`create_worktree` raises AFTER the worktree exists, and before it returns.

    The ledger never hears about it, so the directory survives under the
    campaign id and the next attempt at the same campaign fails on a worktree
    nobody remembers creating.
    """

    class FakeTree:
        path = "/mnt/raid0/llm/ak-worktrees/ak-test"

    def _patched(self, holds):
        proof = mock.Mock(holds=holds, differences=("production moved",))
        return [
            mock.patch.object(campaign.worktree, "GitRepo", lambda *a, **k: object()),
            mock.patch.object(campaign.worktree, "resolve_anchor",
                              lambda *a, **k: object()),
            mock.patch.object(campaign.worktree, "create_campaign_worktree",
                              lambda *a, **k: (self.FakeTree(), proof)),
        ]

    def test_a_mutated_production_tree_still_tears_the_worktree_down(self):
        """THE BITE."""
        torn = []
        patches = self._patched(False) + [
            mock.patch.object(campaign.worktree, "teardown_worktree",
                              lambda tree, **k: torn.append(tree)),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        with self.assertRaises(campaign.worktree.ProductionMutated) as raised:
            campaign.HostOps().create_worktree(spec())
        self.assertEqual(len(torn), 1, "the campaign worktree was left on disk")
        self.assertIn("torn down", str(raised.exception))

    def test_a_teardown_that_fails_does_not_replace_the_mutation_report(self):
        """The mutation is the news; the leaked worktree is carried beside it."""
        def explode(*_a, **_k):
            raise RuntimeError("rm -rf refused")

        patches = self._patched(False) + [
            mock.patch.object(campaign.worktree, "teardown_worktree", explode),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        with self.assertRaises(campaign.worktree.ProductionMutated) as raised:
            campaign.HostOps().create_worktree(spec())
        self.assertIn("production tree", str(raised.exception))
        self.assertIn("could not be torn down", str(raised.exception))

    def test_the_compliant_path_tears_nothing_down(self):
        """CONTROL: a worktree that was created correctly must survive."""
        torn = []
        patches = self._patched(True) + [
            mock.patch.object(campaign.worktree, "teardown_worktree",
                              lambda tree, **k: torn.append(tree)),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        tree = campaign.HostOps().create_worktree(spec())
        self.assertIsInstance(tree, self.FakeTree)
        self.assertEqual(torn, [])


class TestThePreflightIsWiredToSomethingThatExists(unittest.TestCase):
    """`device_claim_witness_reader()` takes the ids it is a witness FOR.

    They are required-positional, so the call raised `TypeError` on the first
    line of every executing run: `--execute` could not get past step 0. Nothing
    caught it because no test has ever constructed a `HostOps` preflight.
    """

    def _patched(self, verdict, *, boosting=117, load1=96.0):
        class FakeRepo:
            def __init__(self, path):
                self.path = path

            @staticmethod
            def commit_parents(_commit):
                return (campaign.PRODUCTION_COMMIT,)

            @staticmethod
            def is_ancestor(_ancestor, _descendant):
                return True

        def resolved(_repo, branch, *, expected_commit):
            return mock.Mock(
                commit=expected_commit,
                fingerprint=mock.Mock(
                    head_commit=expected_commit, symbolic_ref=branch,
                    status_porcelain=""))

        state = mock.Mock(khz_by_cpu=tuple((c, 3_000_000) for c in range(boosting))
                          + tuple((c, 1_000_000) for c in range(boosting, 96)),
                          load1=load1, uptime_s=1000.0, cpu_list="0-95",
                          package_by_cpu=tuple((c, 0) for c in range(96)),
                          package_energy_uj=((0, 1_000_000, 100_000_000,
                                              "/power/package0"),))
        def preflight_result(*_args, **_kwargs):
            result = mock.Mock(verdict=verdict, reasons=("fake",))
            result.as_check.return_value = schemas.Check(verdict, ("fake",))
            result.to_dict.return_value = {
                "verdict": verdict, "basis": "fixture",
                "scope": {"label": "ak-test"}, "findings": [],
                "owned": None, "region_claims": [], "gpu_claims": []}
            return result
        return [
            mock.patch.object(campaign.worktree, "frozen_tree_paths", lambda: ()),
            mock.patch.object(campaign.worktree, "GitRepo", FakeRepo),
            mock.patch.object(campaign.worktree, "resolve_anchor", resolved),
            mock.patch.object(campaign.cpu_region_claim, "verify_host_topology",
                              lambda *a, **k: schemas.Check(schemas.PASS, ())),
            mock.patch.object(campaign.preflight, "preflight",
                              preflight_result),
            mock.patch.object(campaign.microbench, "read_host_state",
                              lambda **k: state),
        ]

    def _run(self, verdict, **kw):
        for patch in self._patched(verdict, **kw):
            patch.start()
            self.addCleanup(patch.stop)
        return campaign.HostOps().preflight(spec(
            journal_root="/mnt/raid0/llm/epyc-inference-research/data/ak-preflight-test"))

    def test_the_preflight_runs_at_all_on_both_cells(self):
        """THE BITE: `HostOps.preflight` raised TypeError on its own third line.

        `device_claim_witness_reader()` was called with no arguments and the
        device ids it is a witness FOR are required-positional. The claim
        sources are built before the (patched) verdict call, so this exercises
        the real wiring on both cells; before the fix it raised.
        """
        for patch in self._patched(schemas.PASS, boosting=16, load1=3.3):
            patch.start()
            self.addCleanup(patch.stop)
        journal_root = "/mnt/raid0/llm/epyc-inference-research/data/ak-preflight-test"
        for built in (spec(journal_root=journal_root),
                      spec(backend="llama_gpu", devices=("ROCm0",),
                           device_names=("AMD Instinct MI210",),
                           journal_root=journal_root)):
            with self.subTest(backend=built.backend):
                check = campaign.HostOps().preflight(built)
                self.assertIn(check.outcome,
                              (schemas.PASS, schemas.COULD_NOT_CHECK, schemas.FAIL))
        self.assertTrue(callable(claim_witness.gpu_claim_sources(()).gpu_claim_reader))

    def test_an_executing_host_path_refuses_without_a_run_ledger_root(self):
        check = campaign.HostOps().preflight(spec())
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("--journal-root", " ".join(check.reasons))

    def test_an_unevaluable_concurrency_check_refuses_the_run(self):
        """FAIL-OPEN, closed: 'I could not tell' must not start a benchmark.

        `preflight.require_no_concurrent_inference` — the module's own
        recommended call site — refuses anything but PASS. Folding
        COULD_NOT_CHECK softly meant an unreadable lock root started a run on a
        host somebody else might be measuring on.
        """
        # A QUIET host — load 3.3 over 96 cores — so the only non-PASS input is
        # the concurrency verdict. Otherwise the load ceiling supplies the FAIL
        # and the test passes without the guard.
        check = self._run(schemas.COULD_NOT_CHECK, boosting=16, load1=3.3)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("concurrent_inference", " ".join(check.reasons))

    def test_an_idle_host_is_still_startable(self):
        """CONTROL, and it is the one that matters: the boost check's own
        COULD_NOT_CHECK is the NORMAL reading before a run, and hardening the
        concurrency layer must not resurrect the idle-frequency trap."""
        check = self._run(schemas.PASS, boosting=16, load1=3.3)
        self.assertNotEqual(check.outcome, schemas.FAIL)
        self.assertIn("IDLE", " ".join(check.reasons))

    def test_the_boost_gate_cannot_rule_in_a_preflight_and_the_run_proceeds(self):
        """PINS A KNOWN LIMIT rather than asserting a property that cannot hold.

        `LOADED_ENOUGH_TO_JUDGE_BOOST` and `HostStatePolicy.max_load_per_core`
        are THE SAME NUMBER with opposite senses: below 0.25/core the boost
        count is declared unevaluable, and above 0.25/core `check_load` refuses
        the run outright. The only load at which the boost gate can PASS is
        exactly 0.25/core. So a `HostOps` preflight never returns PASS, and the
        `if outcome == PASS` branch at the end of it is unreachable.

        That is not a fail-open — the run still proceeds, and the clock is
        judged where it is valid — but it must be recorded rather than believed
        away, and the preflight's record carries the reading either way.
        """
        quiet = self._run(schemas.PASS, boosting=16, load1=3.3)
        self.assertEqual(quiet.outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(campaign.LOADED_ENOUGH_TO_JUDGE_BOOST,
                         campaign.microbench.HostStatePolicy().max_load_per_core)

    def test_a_loaded_host_is_refused_by_the_load_ceiling(self):
        """CONTROL, on the other side of the same threshold: the guard bites."""
        check = self._run(schemas.PASS, boosting=90, load1=96.0)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("contention", " ".join(check.reasons))

    def test_a_cpu_campaign_refuses_before_claim_when_package_power_is_unreadable(self):
        patches = self._patched(schemas.PASS, boosting=16, load1=3.3)
        state = mock.Mock(
            khz_by_cpu=tuple((c, 3_000_000) for c in range(16))
                        + tuple((c, 1_000_000) for c in range(16, 96)),
            load1=3.3, uptime_s=1000.0, cpu_list="0-95",
            package_by_cpu=tuple((c, 0) for c in range(96)),
            package_energy_uj=())
        patches[-1] = mock.patch.object(
            campaign.microbench, "read_host_state", lambda **k: state)
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        check = campaign.HostOps().preflight(spec(
            journal_root="/mnt/raid0/llm/epyc-inference-research/data/ak-preflight-test"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("package_power_available", " ".join(check.reasons))


# =============================================================================
# 8b. The order draw — `Pair.order` was recorded and never read
# =============================================================================

class TestTheOrderDrawMustNotBeDegenerate(unittest.TestCase):
    """`statistics.OrderSchedule` is a COIN FLIP PER BLOCK, not an alternation.

    `_base_order()` hashes `(campaign_seed, "order", candidate_id, index)` and
    takes the parity, so five blocks land all one way once in sixteen runs. The
    accept rule carried `Pair.order` into its record and never looked at it,
    which made the field documentation: a 5-0 run is a sequential A/B, and the
    two existing controls cannot see it —

      * the anchor-arm control reads BETWEEN blocks, and a host that boosts at
        each block's start and sags inside it leaves the anchor series flat;
      * the drift bound is a between-run quantity for the same reason.

    In a 5-0 run the within-block slot effect is perfectly confounded with the
    candidate effect, so it is refused in BOTH directions rather than in the
    one that happens to flatter.
    """

    BOUND = campaign.DRIFT_BOUND_BY_METRIC["decode_tokens_per_s"]

    def decide(self, pairs):
        return campaign.decide(pairs, t0=PASSING_T0, blocks_precommitted=5,
                               drift_bound=self.BOUND)

    #: A within-block sag with NO between-block trend: each block starts at 52.0
    #: and drops 4% inside itself, then the next block starts at 52.0 again.
    #: This is boost behaviour, and it is invisible to every between-block
    #: control in the rule.
    @staticmethod
    def sawtooth(orders):
        out = []
        for i, order in enumerate(orders):
            fast, slow = 52.0, 52.0 * 0.96
            if order == "candidate_first":
                candidate, anchor = fast, slow
            else:
                anchor, candidate = fast, slow
            out.append(campaign.Pair(block_index=i, anchor=anchor, candidate=candidate,
                                     order=order))
        return tuple(out)

    def test_a_five_zero_draw_that_flatters_the_candidate_is_refused(self):
        """THE BITE. Identical code, 5-0 candidate-first, a manufactured +4.2%.

        Every conjunct of the accept rule passes: every pair favours the
        candidate, the median relative gain is above the drift bound, and the
        anchor arm did not move at all between blocks. Before the order control
        this KEPT a null.
        """
        pairs = self.sawtooth(("candidate_first",) * 5)
        decision = self.decide(pairs)
        self.assertFalse(decision.keep, decision.reason)
        self.assertIn("inadmissible", decision.reason)
        # ...and it was NOT caught by either existing control.
        self.assertGreater(decision.min_delta, 0)
        self.assertGreater(decision.median_relative, self.BOUND)
        self.assertAlmostEqual(decision.anchor_drift, 0.0, places=12)

    def test_a_five_zero_draw_the_other_way_is_refused_too(self):
        """Both directions. Which way it runs is the number this design cannot
        measure — it is confounded with the effect — so 'it only handicapped the
        candidate' is an assumption, not a reading."""
        pairs = self.sawtooth(("anchor_first",) * 5)
        decision = self.decide(pairs)
        self.assertFalse(decision.keep)
        self.assertIn("inadmissible", decision.reason)

    def test_a_four_one_draw_is_admitted(self):
        """THE COMPLIANT-PATH CONTROL, and it must be the lopsided one.

        A guard that refused every imbalance would discard 37.5% of five-block
        runs after spending the whole claim window. Only the degenerate draw —
        where NO block ever gave the other arm the earlier slot — is refused.
        """
        pairs = pairs_from_positions(
            TG128_OVER_TEN_POSITIONS, candidate_factor=1.08,
            orders=("candidate_first",) * 4 + ("anchor_first",))
        decision = self.decide(pairs)
        self.assertTrue(decision.keep, decision.reason)

    def test_the_orders_are_published_in_the_record(self):
        """A number the rule read must be in the record it wrote."""
        pairs = pairs_from_positions(TG128_OVER_TEN_POSITIONS, candidate_factor=1.08,
                                     orders=BALANCED)
        decision = self.decide(pairs)
        self.assertEqual(decision.orders, BALANCED)
        self.assertEqual(decision.to_dict()["orders"], list(BALANCED))

    def test_the_remedy_named_is_not_the_one_that_does_not_work(self):
        """`retry()` flips every element: a 5-0 draw becomes the mirror 5-0."""
        pairs = self.sawtooth(("candidate_first",) * 5)
        reason = self.decide(pairs).reason
        self.assertIn("fresh campaign seed", reason)
        self.assertIn("does NOT fix this", reason)


# =============================================================================
# 8c. The boost floor is a RATIO of the claimed footprint
# =============================================================================

class TestTheBoostFloorScalesToTheFootprint(unittest.TestCase):
    """`80 # of 96` is a count AND a denominator. The GPU cell pins eight cores.

    Applied verbatim to `184-191`, "80 boosting" is unreachable by a perfectly
    healthy MI210 host, so the preflight FAILed the compliant path — the shape
    `feedback_guard_must_not_forbid_its_own_idiom` is about, and the first thing
    anyone does with a gate like that is switch it off.
    """

    def test_the_gpu_cells_eight_cores_can_pass(self):
        """THE BITE. Eight of eight boosting under load is a healthy host."""
        check = campaign.check_boost_under_load(boosting_cores=8, load1=8.0, cpu_count=8)
        self.assertEqual(check.outcome, schemas.PASS, " ".join(check.reasons))

    def test_a_throttled_eight_core_footprint_still_FAILS(self):
        """The guard must still bite on the footprint it was rescaled for."""
        check = campaign.check_boost_under_load(boosting_cores=3, load1=8.0, cpu_count=8)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("poisoned", " ".join(check.reasons))

    def test_the_canonical_footprint_is_unchanged_where_it_was_ratified(self):
        self.assertEqual(campaign.required_boosting_cores(96), campaign.BOOST_MIN_CORES)
        self.assertEqual(campaign.required_boosting_cores(192), campaign.BOOST_MIN_CORES)

    def test_the_floor_is_the_ratified_ratio(self):
        self.assertEqual(campaign.required_boosting_cores(8), 7)   # ceil(8 * 80/96)
        self.assertEqual(campaign.required_boosting_cores(1), 1)

    def test_a_footprint_of_no_cores_is_not_a_footprint(self):
        with self.assertRaises(ValueError):
            campaign.required_boosting_cores(0)

    def test_the_gpu_cell_reports_its_own_footprint(self):
        built = spec(backend="llama_gpu", devices=("ROCm0",),
                     device_names=("AMD Instinct MI210",))
        self.assertEqual(built.cpu_list, "184-191")
        self.assertEqual(built.cpu_count, 8)
        self.assertEqual(spec().cpu_count, 96)


# =============================================================================
# 9. The boundary — enforced against this module's AST
# =============================================================================

#: The package's own absolute prefix. An import that spells it out reaches the
#: same modules a relative one does, and a walker that only understands `from .`
#: is a boundary anyone can step over by typing more.
PACKAGE_PREFIX = "scripts.kernel_rnd.autokernel."


def _relative(dotted: str) -> str:
    """`scripts.kernel_rnd.autokernel.controller.selection` -> `controller.selection`."""
    for prefix in (PACKAGE_PREFIX, "autokernel."):
        if dotted.startswith(prefix):
            return dotted[len(prefix):]
    return dotted


def internal_imports_in(source: str) -> set:
    """Every package-internal module `source` imports, as package-relative names.

    `ast.walk`, so an import nested inside a function or an `if` is found too —
    a lazy import is the obvious way around a boundary that only reads the top
    of the file. Absolute spellings are normalized to the relative ones the
    boundary tables use, for the same reason.
    """
    tree = ast.parse(source)
    found: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                for alias in node.names:
                    found.add(f"{base}.{alias.name}" if base else alias.name)
            elif base.startswith(PACKAGE_PREFIX) or base.startswith("autokernel."):
                relative = _relative(base)
                for alias in node.names:
                    found.add(f"{relative}.{alias.name}" if relative else alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE_PREFIX) \
                        or alias.name.startswith("autokernel") \
                        or alias.name.startswith("."):
                    found.add(_relative(alias.name))
    return found


def internal_imports(path: Path) -> set:
    return internal_imports_in(path.read_text(encoding="utf-8"))


class TestTheBoundaryIsStructural(unittest.TestCase):
    """The boundary is a test, not a paragraph. Deletion stays the operator's call.

    This is the whole mechanism by which "what is essential" becomes checkable:
    nothing is deleted, nothing is moved, and the driver's own import list is
    pinned against the declared one in both directions.
    """

    PATH = Path(campaign.__file__)

    def test_the_driver_imports_exactly_what_it_declares(self):
        self.assertEqual(internal_imports(self.PATH),
                         set(campaign.MODULES_THE_DRIVER_USES),
                         "campaign.py's imports and MODULES_THE_DRIVER_USES disagree")

    def test_no_over_engineered_subsystem_is_imported(self):
        imported = internal_imports(self.PATH)
        for forbidden in campaign.MODULES_DELIBERATELY_NOT_USED:
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, imported)
                if "." not in forbidden:
                    # A whole SUBPACKAGE is out: nothing under it may be reached
                    # either. (A dotted entry names one module of a package the
                    # driver does use, so the prefix sweep does not apply.)
                    self.assertFalse(
                        any(name.startswith(forbidden + ".") for name in imported),
                        f"campaign.py reaches into {forbidden}")

    def test_every_declared_module_is_real(self):
        """The boundary must not drift into naming modules that do not exist."""
        root = self.PATH.parent
        for declared in (set(campaign.MODULES_THE_DRIVER_USES)
                         | set(campaign.MODULES_DELIBERATELY_NOT_USED)):
            with self.subTest(module=declared):
                stem = root.joinpath(*declared.split("."))
                self.assertTrue(stem.is_dir() or stem.with_suffix(".py").exists(),
                                f"{declared} names nothing on disk")

    def test_every_declared_module_states_why(self):
        for table in (campaign.MODULES_THE_DRIVER_USES,
                      campaign.MODULES_DELIBERATELY_NOT_USED):
            for name, reason in table.items():
                with self.subTest(module=name):
                    self.assertGreater(len(reason), 30,
                                       f"{name} is on the boundary with no argument")

    def test_the_two_tables_do_not_overlap(self):
        self.assertEqual(set(campaign.MODULES_THE_DRIVER_USES)
                         & set(campaign.MODULES_DELIBERATELY_NOT_USED), set())

    # -- the two boundary DECLARATIONS, pinned against each other --------------
    #
    # There are two of them, and they are written in different files for good
    # reasons: `campaign.py` declares what the DRIVER imports (checked against its
    # own AST), `test_campaign_footprint.py` declares what the CLOSURE may reach
    # (checked by walking it). Until 2026-08-04 nothing compared them directly —
    # agreement was enforced only transitively, by both being true of the same
    # tree. That is the shape this package has been burned by twice: two spellings
    # of one fact, and the one that can disagree is the one nobody reads. These
    # two assertions are the direct comparison.

    @staticmethod
    def _footprint_tables():
        """The other declaration, by import. `test_campaign_footprint` deliberately
        never imports the package it guards, so the comparison lives on this side,
        where `campaign` is already imported."""
        from . import test_campaign_footprint as fp
        prefix = f"{fp.ROOT_PKG}."
        deferred = {p[len(prefix):] for p in fp.DEFERRED if p.startswith(prefix)}
        allowed = {m[len(prefix):] for m in fp.CONTROLLER_ALLOWED if m.startswith(prefix)}
        return deferred, allowed

    def test_nothing_the_driver_imports_is_banned_by_the_other_declaration(self):
        """A module in `MODULES_THE_DRIVER_USES` that the closure walk still bans
        is a boundary at war with itself, and the resolution is always to silence
        one of the two rather than to look at which is right."""
        deferred, allowed = self._footprint_tables()
        conflicting = sorted(
            name for name in campaign.MODULES_THE_DRIVER_USES
            if name not in allowed
            and any(name == d or name.startswith(d + ".") for d in deferred))
        self.assertEqual(
            conflicting, [],
            f"campaign.py imports {conflicting}, which test_campaign_footprint.DEFERRED "
            "bans and CONTROLLER_ALLOWED does not except")

    def test_nothing_the_driver_disowns_is_excepted_by_the_other_declaration(self):
        """The mirror: an allow-list row for a module this file says is not used."""
        _deferred, allowed = self._footprint_tables()
        contradicted = sorted(set(campaign.MODULES_DELIBERATELY_NOT_USED) & allowed)
        self.assertEqual(
            contradicted, [],
            f"{contradicted} is both deliberately-not-used here and allow-listed there")

    def test_no_e_process_is_reachable_from_the_driver(self):
        """The measured 1.6-1.9% CV does not justify one, and it was UNPASSABLE.

        Scanned over IDENTIFIERS, not over the file's text: naming the extension
        round in a docstring to say why it is not run is exactly the argument
        this file is supposed to carry, and a text scan would forbid it.
        """
        tree = ast.parse(self.PATH.read_text(encoding="utf-8"))
        names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        for forbidden in ("e_value", "sequential_evaluation", "ExtensionAuthorization",
                          "solve_mde", "PairedBlockReducer", "reduce_blocks"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, names)

    def test_integrity_is_reached_exactly_once_and_through_the_seam(self):
        """`chain` owns the projection; a second consumer is a second derivation."""
        source = self.PATH.read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if "chain.integrity" in line and not line.strip().startswith("#"))
        self.assertEqual(code.count("chain.integrity"), 1, code)
        self.assertIn("hash_source_tree", code)

    def test_no_forbidden_module_is_reached_through_an_allowed_alias(self):
        """A re-export is an import with extra steps. `chain.surface_module` is
        right there, and `microbench.statistics` is too: every module the driver
        imports carries the ones IT imports as attributes, so a boundary that
        reads `import` statements only is a boundary over one spelling.

        The single licensed reach is `chain.integrity`, which the driver argues
        for by name and the next test pins to exactly one call.
        """
        tree = ast.parse(self.PATH.read_text(encoding="utf-8"))
        aliases = {"schemas", "storage", "journal_module", "api", "correctness", "devices",
                   "recipes", "chain", "cpu_region_claim", "microbench", "physical_bounds",
                   "t0_provider",
                   "worktree", "claim_witness", "device_claim", "preflight",
                   # 2026-08-04: the hypothesis plane. Every module the driver
                   # imports carries the ones IT imports as attributes, and these
                   # two are no exception — `hypotheses.schemas` resolves.
                   "hypotheses", "do_not_repeat"}
        forbidden_tails = {name.split(".")[-1]
                           for name in campaign.MODULES_DELIBERATELY_NOT_USED}
        forbidden_tails |= {"surface_module", "statistics"}
        reached = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in aliases and node.attr in forbidden_tails:
                reached.add(f"{node.value.id}.{node.attr}")
        self.assertEqual(reached, {"chain.integrity"},
                         f"the driver reaches a deferred subsystem through an alias: "
                         f"{sorted(reached - {'chain.integrity'})}")

    def test_only_HostOps_can_reach_a_spawner(self):
        """A composition pass must not be able to spawn even by mistake.

        `--dry-run` being the default is an argv property; this is the
        structural one. The two things in this package that start a process —
        `microbench.SubprocessSpawner` and `t0_provider.SubprocessRunner` — are
        named inside the `HostOps` class body and nowhere else, so no path
        through `DryRunOps`, `run_campaign` or `main` reaches one.
        """
        tree = ast.parse(self.PATH.read_text(encoding="utf-8"))
        host_ops = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.ClassDef) and node.name == "HostOps")
        inside = {id(node) for node in ast.walk(host_ops)}
        stray = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) \
                    and node.attr in ("SubprocessSpawner", "SubprocessRunner") \
                    and id(node) not in inside:
                stray.append(node.attr)
        self.assertEqual(stray, [], f"a spawner is reachable outside HostOps: {stray}")

    def test_the_driver_imports_nothing_dynamically(self):
        """`importlib.import_module('...controller.selection')` is an import that
        no AST import-walker sees. It is refused by identifier, with the
        compliant idiom (a plain relative import) untouched."""
        tree = ast.parse(self.PATH.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for forbidden in ("importlib", "import_module", "__import__", "exec", "eval"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, names)

    def test_the_driver_spawns_nothing_by_name_pattern(self):
        """INC-20260731. No pkill, no pgrep, no killall, no ps | grep | kill."""
        source = self.PATH.read_text(encoding="utf-8").lower()
        for forbidden in ("pkill", "pgrep", "killall", "os.kill", "shell=true"):
            with self.subTest(token=forbidden):
                # Named in prose is fine; called is not.
                self.assertNotIn(forbidden + "(", source)


class TestTheBoundaryWalkerItself(unittest.TestCase):
    """A guard is only as good as the forms it can see. Each of these is a way
    around an import boundary that reads `from . import x` at the top of a file,
    and each is asserted against the walker on synthetic source — so the guard
    is proven to bite without campaign.py having to contain the violation.
    """

    def test_a_lazy_import_inside_a_function_is_seen(self):
        found = internal_imports_in(
            "def go():\n    from .controller import selection\n    return selection\n")
        self.assertIn("controller.selection", found)

    def test_a_conditional_import_is_seen(self):
        found = internal_imports_in(
            "import os\nif os.environ.get('X'):\n    from . import release\n")
        self.assertIn("release", found)

    def test_an_absolute_import_of_this_package_is_seen(self):
        """THE BITE: `import scripts.kernel_rnd.autokernel.controller.selection`
        reaches exactly what the relative form reaches, and the walker used to
        ignore it because the dotted name starts with 'scripts'."""
        found = internal_imports_in(
            "import scripts.kernel_rnd.autokernel.controller.selection\n")
        self.assertIn("controller.selection", found)

    def test_an_absolute_from_import_is_normalized_too(self):
        found = internal_imports_in(
            "from scripts.kernel_rnd.autokernel.evaluator import statistics\n")
        self.assertIn("evaluator.statistics", found)

    def test_a_try_except_import_is_seen(self):
        found = internal_imports_in(
            "try:\n    from .release import readiness\nexcept ImportError:\n    pass\n")
        self.assertIn("release.readiness", found)

    def test_no_package_dunder_init_smuggles_a_deferred_subsystem(self):
        """The last import route: `from . import schemas` executes
        `autokernel/__init__.py`, and every subpackage's `__init__` runs the
        same way. If one of them imported `controller`, the boundary would be
        decorative no matter what campaign.py says. They are docstring-only.
        """
        root = Path(campaign.__file__).parent
        # Only the `__init__` files the driver's own imports EXECUTE. A deferred
        # subpackage's `__init__` re-exporting its own modules is that package's
        # business — nothing on the driver's path runs it. (`release/__init__`
        # and `surface/__init__` do exactly that, which is why the sweep is
        # scoped rather than global: a test that fails on someone else's file
        # gets deleted, not obeyed.)
        on_the_path = {name.split(".")[0] for name in campaign.MODULES_THE_DRIVER_USES}
        inits = [root / "__init__.py"] + sorted(
            root / package / "__init__.py" for package in on_the_path
            if (root / package).is_dir())
        self.assertGreater(len(inits), 3)
        for init in inits:
            with self.subTest(init=str(init.relative_to(root))):
                found = internal_imports_in(init.read_text(encoding="utf-8"))
                forbidden = {name for name in found
                             if any(name == bad or name.startswith(bad + ".")
                                    for bad in campaign.MODULES_DELIBERATELY_NOT_USED)}
                self.assertEqual(forbidden, set(),
                                 f"{init} smuggles {sorted(forbidden)} onto the path")

    def test_an_unrelated_third_party_import_is_not_a_finding(self):
        """CONTROL: the walker must not forbid its own idiom. `json` and
        `scripts.lib.canonical_recipe` are not this package."""
        found = internal_imports_in(
            "import json\nfrom scripts.lib import canonical_recipe\n")
        self.assertEqual(found, set())

    def test_the_permitted_imports_are_still_recognised(self):
        """CONTROL: the compliant spelling must survive normalization."""
        found = internal_imports_in("from .evaluator import recipes\nfrom . import storage\n")
        self.assertEqual(found, {"evaluator.recipes", "storage"})


class TestTheEntrypointExists(unittest.TestCase):
    """The one thing whose absence is why this package produced nothing."""

    def test_the_module_has_a_main(self):
        self.assertTrue(callable(campaign.main))

    def test_the_module_has_a_dunder_main_guard(self):
        source = Path(campaign.__file__).read_text(encoding="utf-8")
        self.assertIn('if __name__ == "__main__":', source)

    def test_the_parser_offers_the_four_declared_flags(self):
        parser = campaign.build_parser()
        options = {action.dest for action in parser._actions}
        for flag in ("dry_run", "candidate_ref", "backend", "blocks", "campaign_id"):
            self.assertIn(flag, options)

    def test_the_parser_help_does_not_raise(self):
        self.assertIsInstance(campaign.build_parser(), argparse.ArgumentParser)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestTheDryRunShowsPerArmLinkage(unittest.TestCase):
    """The dry run must show that each arm loads its OWN build's libraries.

    Found 2026-08-04 by reading a real dry run rather than a test: step 7 rendered
    a single `env` key, taken from the anchor. The mechanism was already correct —
    `_render_arms` binds `library_path` per arm — but the OUTPUT could not
    distinguish that from both arms sharing production's libs.

    That distinction is the whole review. A candidate linked against the anchor's
    `libggml` measures the anchor whatever it changed, and reports a clean,
    well-formed null: min(delta) 0, median 0, no crossing, REVERT. Every gate
    above it passes, because nothing is wrong except that the wrong binary ran.
    The three source trees run three different ggml generations for this reason,
    and `verify_ggml_linkage.sh` exists because a binary that inherits another
    tree's ggml runs silently wrong.
    """

    def _blocks_step(self):
        out = io.StringIO()
        ops = campaign.DryRunOps(out=out)
        campaign.run_campaign(spec(), ops)
        for step in ops.steps:
            if step.name == "paired_blocks":
                return step.detail
        self.fail("no paired_blocks step in the dry run")

    def test_both_arm_envs_are_rendered(self):
        step = self._blocks_step()
        self.assertIn("anchor_env", step)
        self.assertIn("candidate_env", step)
        self.assertNotIn("env", step,
                         "a single `env` key cannot say which arm it belongs to")

    def test_the_two_arms_do_not_share_a_library_path(self):
        step = self._blocks_step()
        anchor = step["anchor_env"]["LD_LIBRARY_PATH"]
        candidate = step["candidate_env"]["LD_LIBRARY_PATH"]
        self.assertNotEqual(
            anchor, candidate,
            "both arms would load the same libggml; the candidate would measure "
            "the anchor and report a well-formed null")
        self.assertNotIn(campaign.PRODUCTION_REPO,
                         candidate.split(os.pathsep)[0],
                         "the candidate's FIRST library path is production's")


# =============================================================================
# 7. The falsifier-before-compute gate — `--hypothesis`
# =============================================================================
#
# THE DEFECT. `controller/hypotheses.py::claim_for_hypothesis` documents itself
# as "The ONLY route from a hypothesis to a resource claim" and enforces the rule
# that a falsifier is optional when a question is written and MANDATORY before a
# claim is spent on it. It had ZERO non-test callers: `HostOps.acquire_claim`
# called `cpu_region_claim.acquire_cpu_region_claim` directly, so the gate
# enforced nothing — a guard defined and never wired, the fifth of that shape in
# this package.
#
# Every test below drives `main()`, because "refused" is a statement about the
# ENTRYPOINT, not about a helper: the whole point is that the refusal lands
# before a claim is acquired and before a worktree exists. Nothing here spawns,
# claims or builds; the only files written are under a temp tree that is removed.

#: Not `/tmp`: `storage.assert_not_scratch` refuses a scratch journal root (the
#: 2026-07-04 win was written to `/mnt/raid0/llm/tmp/` and that directory no
#: longer exists), and a hypothesis authorization is a durable record. This is
#: the same non-scratch root `execution/test_execution_chain.py` uses.
HYPOTHESIS_SCRATCH_ROOT = "/mnt/raid0/llm/.scratch"

#: An operator entry with NO falsifier. Legal — that is the 2026-08-04 amendment,
#: and the reason the barrier moved to the claim rather than to the entry.
ENTRY_NO_FALSIFIER = {
    "hypothesis_id": "akh-test-absent",
    "statement": "the elementwise/norm cluster is where the B=128 decode time goes",
}

#: The same operator, typing the thing HYPOTHESES.md tells them not to type.
ENTRY_PLACEHOLDER = {
    "hypothesis_id": "akh-test-placeholder",
    "statement": "fusing the norm cluster should be worth 15%",
    "falsifier": "tbd",
}

ENTRY_STATED = {
    "hypothesis_id": "akh-test-stated",
    "statement": "fusing the elementwise/norm cluster lands >= 15% at B=128",
    "falsifier": "a current wall-share map shows the cluster under 20%",
    "regime": {"backend": "llama_cpu", "phase": "decode"},
}


class _HypothesisGateCase(unittest.TestCase):
    """One temp record per test, and a snapshot of everything a spend would touch."""

    def setUp(self) -> None:
        os.makedirs(HYPOTHESIS_SCRATCH_ROOT, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(prefix="ak-hypothesis-",
                                                dir=HYPOTHESIS_SCRATCH_ROOT)
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.join(self._tmp.name, "record")
        os.makedirs(self.root)
        self.campaign_id = "ak-hypothesis-test"
        # What a spend would touch, BEFORE anything runs. `refused` has to mean
        # "nothing was acquired", not "the exit code was 2".
        self.claim_journal = spec().claim_journal_path
        self.worktree_path = spec(campaign_id=self.campaign_id).worktree_path
        self.claim_journal_before = self._claim_journal_state()

    def _claim_journal_state(self):
        """(exists, size, mtime_ns) of the region-claim journal. Never its bytes:
        it is a shared file and another session may legitimately be appending to
        it — what this asserts is that THIS process did not."""
        try:
            st = os.stat(self.claim_journal)
        except FileNotFoundError:
            return (False, 0, 0)
        return (True, st.st_size, st.st_mtime_ns)

    def store(self, *entries) -> str:
        path = os.path.join(self._tmp.name, f"store-{len(entries)}-{id(entries)}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema": "epyc.autokernel.operator_hypotheses.v1",
                       "hypotheses": list(entries)}, handle, indent=2)
        return path

    def run_main(self, *extra, ops=None):
        out, err = io.StringIO(), io.StringIO()
        argv = ["--campaign-id", self.campaign_id, "--candidate-id", "akc-hypothesis",
                "--model", MODEL, "--journal-root", self.root, *extra]
        with contextlib.redirect_stderr(err):
            code = campaign.main(argv, out=out, ops=ops)
        return code, out.getvalue(), err.getvalue()

    def assert_nothing_was_acquired(self, ops) -> None:
        """The assertion the whole gate exists for. A refusal that still spent a
        claim is the defect, not the fix."""
        self.assertEqual(ops.calls, [],
                         f"the loop ran despite the refusal: {ops.calls}")
        self.assertEqual(self._claim_journal_state(), self.claim_journal_before,
                         f"{self.claim_journal} was written to by a REFUSED campaign")
        self.assertFalse(os.path.exists(self.worktree_path),
                         f"{self.worktree_path} exists; the refusal came too late")


class TestAHypothesisWithNoFalsifierCannotReachAClaim(_HypothesisGateCase):
    """§8.4.0: optional on entry, mandatory before compute. THE BITE."""

    def test_it_is_refused_and_nothing_is_acquired(self):
        ops = SpyOps()
        code, _out, err = self.run_main(
            "--hypothesis", "akh-test-absent",
            "--hypothesis-store", self.store(ENTRY_NO_FALSIFIER), ops=ops)
        self.assertEqual(code, 2, err)
        self.assertIn("FalsifierRequiredBeforeCompute", err)
        self.assertIn("'absent'", err)
        self.assert_nothing_was_acquired(ops)

    def test_the_refusal_says_what_to_do_about_it(self):
        """A refusal that does not name the way out gets worked around."""
        _code, _out, err = self.run_main(
            "--hypothesis", "akh-test-absent",
            "--hypothesis-store", self.store(ENTRY_NO_FALSIFIER), ops=SpyOps())
        self.assertIn("propose_falsifier()", err)


class TestAPlaceholderFalsifierCannotReachAClaimEither(_HypothesisGateCase):
    """'tbd' is an empty string wearing a hat, and it is a DIFFERENT state.

    Absent and placeholder are distinct all the way down — different refusal
    types, different messages, different remedies — and collapsing them is a
    defect the hypothesis work specifically closed. A single merged refusal
    would prove only that something was rejected.
    """

    def test_it_is_refused_and_nothing_is_acquired(self):
        ops = SpyOps()
        code, _out, err = self.run_main(
            "--hypothesis", "akh-test-placeholder",
            "--hypothesis-store", self.store(ENTRY_PLACEHOLDER), ops=ops)
        self.assertEqual(code, 2, err)
        self.assertIn("placeholder", err)
        self.assert_nothing_was_acquired(ops)

    def test_absent_and_placeholder_are_not_the_same_refusal(self):
        """THE BITE: two states, two refusal TYPES, and two different remedies.

        Not asserted by "the word 'absent' appears in one and not the other" —
        the placeholder refusal names the absent state deliberately, to say that
        leaving the field out would have been legal and typing 'tbd' is not.
        What distinguishes them is the exception each raises and the way out each
        offers, and that is what is pinned.
        """
        _c1, _o1, absent = self.run_main(
            "--hypothesis", "akh-test-absent",
            "--hypothesis-store", self.store(ENTRY_NO_FALSIFIER), ops=SpyOps())
        _c2, _o2, placeholder = self.run_main(
            "--hypothesis", "akh-test-placeholder",
            "--hypothesis-store", self.store(ENTRY_PLACEHOLDER), ops=SpyOps())
        self.assertNotEqual(absent, placeholder)
        self.assertIn("FalsifierRequiredBeforeCompute", absent)
        self.assertNotIn("FalsifierRequiredBeforeCompute", placeholder)
        self.assertIn("FalsifierMissing", placeholder)
        self.assertNotIn("FalsifierMissing", absent)
        # The remedies differ, which is the practical half of the distinction:
        # write the predicate, versus stop writing a hat.
        self.assertIn("propose_falsifier()", absent)
        self.assertNotIn("propose_falsifier()", placeholder)
        self.assertIn("is a placeholder", placeholder)
        self.assertNotIn("is a placeholder", absent)


class TestAStatedFalsifierReachesTheClaimAndTravelsWithIt(_HypothesisGateCase):
    """The COMPLIANT PATH for the gate: a real falsifier proceeds.

    Without this, a gate that refused every hypothesis — including every
    legitimate one — would look like the strongest enforcement in the file.
    """

    def compose(self):
        ops = campaign.DryRunOps(out=io.StringIO())
        code, out, err = self.run_main(
            "--hypothesis", "akh-test-stated",
            "--hypothesis-store", self.store(ENTRY_STATED), ops=ops)
        return code, out, err, ops

    def claim_step(self, ops):
        for step in ops.steps:
            if step.name == "acquire_claim":
                return step.detail
        self.fail(f"no acquire_claim step was composed: {ops.calls}")

    def test_the_campaign_proceeds(self):
        code, _out, err, ops = self.compose()
        self.assertEqual(code, 0, err)
        self.assertIn("acquire_claim", ops.calls)

    def test_the_claims_own_purpose_carries_the_falsifier(self):
        """THE BITE, and it is why `claim_for_hypothesis` takes `purpose` OFF the
        token rather than from the caller: the resource record and the question
        record then say the same thing without anyone keeping them in step."""
        _code, _out, _err, ops = self.compose()
        purpose = self.claim_step(ops)["purpose"]
        self.assertIn(ENTRY_STATED["falsifier"], purpose)
        self.assertIn("akh-test-stated", purpose)
        self.assertIn("stated_with_the_hypothesis", purpose)

    def test_the_banner_names_the_question(self):
        _code, out, _err, _ops = self.compose()
        self.assertIn("akh-test-stated", out)
        self.assertIn(ENTRY_STATED["falsifier"], out)

    def test_a_caller_supplied_purpose_is_refused_by_the_door(self):
        """The gate's own rule, asserted from this side of the seam: the driver
        must not be able to write the claim's purpose when a token exists."""
        token = campaign.authorize_for(
            spec(campaign_id=self.campaign_id, journal_root=self.root),
            "akh-test-stated", store_path=self.store(ENTRY_STATED))
        with self.assertRaises(ValueError):
            campaign.hypotheses.claim_for_hypothesis(
                token, lambda **kw: kw, purpose="something else")


class TestNoHypothesisIsExploratoryAndSaysSo(_HypothesisGateCase):
    """COMPLIANT PATH: the default is unchanged, and the record is not silent."""

    def compose(self, *extra):
        ops = campaign.DryRunOps(out=io.StringIO())
        code, out, err = self.run_main(*extra, ops=ops)
        return code, out, err, ops

    def test_a_campaign_without_a_hypothesis_still_works(self):
        code, _out, err, ops = self.compose()
        self.assertEqual(code, 0, err)
        self.assertEqual(
            ops.calls,
            ["preflight", "acquire_claim", "create_worktree", "apply_candidate",
             "build", "t0", "paired_blocks", "keep_or_revert", "teardown_worktree",
             "release_claim", "prove_production_unchanged", "journal"])

    def test_the_record_says_exploratory_rather_than_saying_nothing(self):
        """THE BITE. An unexplained absence and a declared exploratory run must
        not read the same afterwards — the same discipline
        `ClaimAuthorization.do_not_repeat_outcome` applies one field over, where
        a defaulted verdict would make "we asked" and "we never wired it up"
        indistinguishable."""
        record = spec().to_dict()["hypothesis"]
        self.assertFalse(record["bound"])
        self.assertIsNone(record["hypothesis_id"])
        self.assertIn("EXPLORATORY", record["note"])
        self.assertEqual(record["note"], campaign.EXPLORATORY_NOTE)

    def test_the_claim_purpose_declares_it_too(self):
        """The RESOURCE record, not only the campaign record: a claim journal
        entry that just said 'AutoKernel campaign …' cannot be told apart from
        one whose hypothesis binding was dropped on the floor."""
        _code, _out, _err, ops = self.compose()
        purpose = next(s.detail["purpose"] for s in ops.steps
                       if s.name == "acquire_claim")
        self.assertIn("exploratory", purpose)
        self.assertNotIn("[hypothesis", purpose)

    def test_the_banner_says_it_resolves_no_question(self):
        _code, out, _err, _ops = self.compose()
        self.assertIn("EXPLORATORY", out)


class TestAnUnknownHypothesisIsRefusedNotIgnored(_HypothesisGateCase):
    """A typo must not silently downgrade a bound campaign to an exploratory one.

    That is the fail-open direction: the run happens, the record says
    "exploratory", and nobody learns that the question they meant to spend a
    claim on was never asked.
    """

    def test_it_is_refused(self):
        ops = SpyOps()
        code, _out, err = self.run_main(
            "--hypothesis", "akh-does-not-exist",
            "--hypothesis-store", self.store(ENTRY_STATED), ops=ops)
        self.assertEqual(code, 2, err)
        self.assertIn("UnknownHypothesis", err)
        self.assert_nothing_was_acquired(ops)

    def test_it_is_not_treated_as_no_hypothesis_at_all(self):
        _code, out, err = self.run_main(
            "--hypothesis", "akh-does-not-exist",
            "--hypothesis-store", self.store(ENTRY_STATED), ops=SpyOps())
        self.assertNotIn("EXPLORATORY", out)
        self.assertNotIn("state:", out)

    def test_a_hypothesis_without_a_record_root_is_refused_before_anything(self):
        """`--hypothesis` needs somewhere to write the authorization down.

        Refused at the door rather than defaulted to a root nobody declared: an
        authorization whose ledger the next session cannot find is an
        authorization with nothing behind it.
        """
        ops = SpyOps()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err):
            code = campaign.main(
                ["--campaign-id", self.campaign_id, "--candidate-id", "akc-hypothesis",
                 "--model", MODEL, "--hypothesis", "akh-test-stated"],
                out=out, ops=ops)
        self.assertEqual(code, 2, err.getvalue())
        self.assertIn("--journal-root", err.getvalue())
        self.assert_nothing_was_acquired(ops)


class TestTheGateIsTheSameGateInBothModes(_HypothesisGateCase):
    """A gate that only runs under `--execute` is a gate wired to the one mode no
    test can exercise — which is the shape of the defect being fixed."""

    def test_the_dry_run_authorization_says_it_was_a_dry_run(self):
        token = campaign.authorize_for(
            spec(campaign_id=self.campaign_id, journal_root=self.root),
            "akh-test-stated", store_path=self.store(ENTRY_STATED), dry_run=True)
        self.assertIn("DRY RUN", token.purpose)
        self.assertIn("no claim was spent", token.purpose)

    def test_an_executing_authorization_does_not(self):
        """CONTROL: the marker is a statement about the run, not decoration."""
        token = campaign.authorize_for(
            spec(campaign_id=self.campaign_id, journal_root=self.root),
            "akh-test-stated", store_path=self.store(ENTRY_STATED), dry_run=False)
        self.assertNotIn("DRY RUN", token.purpose)

    def test_execute_after_dry_run_reuses_the_open_question(self):
        """Composition may precede execution under the exact campaign identity.

        Intake remains idempotent, while the ledger honestly records two distinct
        authorizations: one composed-only token and one live-spend token.
        """
        store = self.store(ENTRY_STATED)
        run_spec = spec(campaign_id=self.campaign_id, journal_root=self.root)
        composed = campaign.authorize_for(
            run_spec, "akh-test-stated", store_path=store, dry_run=True)
        executing = campaign.authorize_for(
            run_spec, "akh-test-stated", store_path=store, dry_run=False)
        self.assertLess(composed.ledger_seq, executing.ledger_seq)
        self.assertIn("DRY RUN", composed.purpose)
        self.assertNotIn("DRY RUN", executing.purpose)
        events = campaign.hypotheses.HypothesisLedger(
            os.path.join(self.root, campaign.hypotheses.LEDGER_FILENAME)).read().events
        self.assertEqual(sum(event.kind == campaign.hypotheses.EVENT_OPENED
                             for event in events), 1)
        self.assertEqual(sum(event.kind == campaign.hypotheses.EVENT_CLAIM_AUTHORIZED
                             for event in events), 2)

    def test_the_authorization_is_a_durable_record(self):
        """`authorize_claim` writes a CLAIM_AUTHORIZED event before it returns a
        token; a token with no record behind it is not an authorization."""
        campaign.authorize_for(
            spec(campaign_id=self.campaign_id, journal_root=self.root),
            "akh-test-stated", store_path=self.store(ENTRY_STATED))
        ledger = os.path.join(self.root, "hypotheses.jsonl")
        self.assertTrue(os.path.exists(ledger))
        with open(ledger, encoding="utf-8") as handle:
            kinds = [json.loads(line)["kind"] for line in handle if line.strip()]
        self.assertIn("HYPOTHESIS_CLAIM_AUTHORIZED", kinds)

    def test_a_token_from_another_campaign_is_refused_by_the_spec(self):
        """A capability that travelled between campaigns would charge this run's
        claim to another run's question."""
        token = campaign.authorize_for(
            spec(campaign_id=self.campaign_id, journal_root=self.root),
            "akh-test-stated", store_path=self.store(ENTRY_STATED))
        with self.assertRaises(ValueError):
            spec(campaign_id="ak-somewhere-else", authorization=token)

    def test_a_string_is_not_an_authorization(self):
        """CONTROL on the type gate: the spec takes a capability, not a name."""
        with self.assertRaises(TypeError):
            spec(authorization="akh-test-stated")


class TestProspectiveEvaluationDurability(unittest.TestCase):
    """An evaluated run writes evaluation evidence before terminal STOP_STATE."""

    def test_t0_failure_caches_evidence_without_materializing_a_candidate(self):
        ops = campaign.HostOps(nominal_khz=1)
        event = {"event_id": "ake-t0-refusal-fixture"}
        with mock.patch.object(ops, "_evaluation_events", return_value=(event,)), \
                mock.patch.object(
                    campaign.candidate_record, "build_candidate_record",
                    side_effect=AssertionError("T0 refusal must not build a candidate")):
            ops.prepare_durable_records(
                spec(proposal=iqk_parameter_proposal()),
                state=campaign.STATE_T0_FAILED, decision=None)
        self.assertEqual(ops._cached_evaluation_events, (event,))
        self.assertIsNone(ops._cached_candidate_record)

    def test_error_caches_evidence_without_deriving_a_candidate(self):
        """The primary error must not be replaced by a CapturePlanError."""
        ops = campaign.HostOps(nominal_khz=1)
        event = {"event_id": "ake-error-fixture"}
        with mock.patch.object(ops, "_evaluation_events", return_value=(event,)), \
                mock.patch.object(
                    campaign.candidate_record, "build_candidate_record",
                    side_effect=AssertionError("error must not derive a candidate")):
            ops.prepare_durable_records(
                spec(proposal=iqk_parameter_proposal()),
                state=campaign.STATE_ERROR, decision=None)
        self.assertEqual(ops._cached_evaluation_events, (event,))
        self.assertIsNone(ops._cached_candidate_record)

    def test_t0_failure_writes_no_speed_and_evaluation_precedes_stop(self):
        ops = EvaluationSpyOps(t0=FAILING_T0)
        result = campaign.run_campaign(spec(), ops)
        self.assertEqual(result.state, campaign.STATE_T0_FAILED)
        self.assertNotIn("run_paired_blocks", ops.calls)
        self.assertLess(ops.calls.index("close_evaluation_window"),
                        ops.calls.index("release_claim"))
        self.assertLess(ops.calls.index("journal_evaluation"), ops.calls.index("journal"))

    def test_close_window_failure_does_not_skip_release_or_terminal_record(self):
        ops = EvaluationSpyOps(t0=FAILING_T0,
                               fail_at="close_evaluation_window")
        result = campaign.run_campaign(spec(), ops)
        self.assertEqual(result.state, campaign.STATE_ERROR)
        self.assertIn("close_evaluation_window", result.error)
        self.assertIn("release_claim", ops.calls)
        self.assertEqual(ops.calls[-1], "journal")

    def test_evaluation_append_failure_is_non_ok_and_stop_is_still_written(self):
        durable_parent = Path(campaign.__file__).resolve().parents[3] / "data"
        durable_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=durable_parent) as root:
            ops = EvaluationSpyOps(
                t0=FAILING_T0, evaluation_raises=True)
            result = campaign.run_campaign(spec(journal_root=root), ops)
            self.assertFalse(result.ok)
            self.assertIn("evaluation append failure", result.journal_error)
            self.assertEqual(ops.calls[-1], "journal")
            self.assertFalse(ops.journaled["result"]["ok"])

    def test_dry_run_has_no_evaluation_event_seam(self):
        ops = campaign.DryRunOps(out=io.StringIO())
        result = campaign.run_campaign(spec(), ops)
        self.assertEqual(result.state, campaign.STATE_COMPOSED)
        self.assertNotIn("journal_evaluation", ops.calls)

    def test_dry_run_writes_zero_candidate_records(self):
        durable_parent = Path(campaign.__file__).resolve().parents[3] / "data"
        durable_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=durable_parent) as root:
            ops = campaign.DryRunOps(out=io.StringIO())
            campaign.run_campaign(spec(journal_root=root), ops)
            self.assertFalse(Path(root, journal_module.BASE_SHARD_NAME).exists())

    def test_idempotent_append_reuses_identical_record_and_refuses_collision(self):
        from .test_journal import _candidate, _event

        durable_parent = Path(campaign.__file__).resolve().parents[3] / "data"
        durable_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=durable_parent) as root:
            run_spec = spec(journal_root=root)
            ops = campaign.HostOps(nominal_khz=1)
            record = _event("live-writer")
            record["campaign_id"] = run_spec.campaign_id
            record["candidate_id"] = run_spec.candidate_id
            candidate = _candidate("live-writer")
            candidate["campaign_id"] = run_spec.campaign_id
            candidate["candidate_id"] = run_spec.candidate_id
            candidate["proposal_id"] = run_spec.proposal_id or "akp-test-0001"
            candidate["evaluation_event_ids"] = [record["event_id"]]
            ops._cached_evaluation_events = (record,)
            ops._cached_candidate_record = candidate
            first = ops.journal_evaluation(run_spec, None)
            second = ops.journal_evaluation(run_spec, None)
            self.assertEqual(first, second)
            book = journal_module.Journal(root, campaign_id=run_spec.campaign_id)
            entries = book.read_all()
            self.assertEqual([entry.kind for entry in entries], [
                journal_module.KIND_EVALUATION_EVENT,
                journal_module.KIND_CANDIDATE_RECORDED])
            mutated = json.loads(json.dumps(record))
            mutated["status"] = "fail"
            ops._cached_evaluation_events = (mutated,)
            with self.assertRaisesRegex(RuntimeError, "different bytes"):
                ops.journal_evaluation(run_spec, None)

    def test_every_pre_behavioural_t0_refusal_emits_a_schema_valid_non_rate_event(self):
        sha = lambda value: __import__("hashlib").sha256(value.encode()).hexdigest()
        controls = campaign.api.CampaignControls(
            calibration_block_count=20, contribution_floor=0.03,
            max_candidates=10, confirmation_admission_count=2,
            max_blocks_per_candidate=20, storage_floor_bytes_free=1)
        calibration = campaign.api.CalibrationOutputs(
            backend="llama_cpu", phase="prefill", cell_class="tiny_real_graph",
            noise_floor_phi=0.03, b_min_blocks=10, alpha_sel=0.1,
            alpha_conf=0.05, anchor_gate_band=(90.0, 110.0), accepted=True,
            solve_order_recorded=campaign.api.CALIBRATION_SOLVE_ORDER,
            samples_ref="sha256:" + sha("calibration"),
            e_process_construction_id="sign_martingale_predictable_lambda/v1")
        anchor = campaign.api.AnchorIdentity(
            source_commit=campaign.PRODUCTION_COMMIT, binary_sha256=sha("anchor-bin"),
            linkage_sha256=sha("anchor-link"), tool="llama-cli")
        early_request = campaign.api.EvaluationRequest(
            event_id="ake-early-refusal", campaign_id="ak-test",
            candidate_id="akc-test", tier="T0", backend="llama_cpu",
            phase="prefill", cell_class="tiny_real_graph",
            protocol_id="P-AK-SEARCH-1/v1",
            artifact=campaign.api.ArtifactIdentity(
                sha("source"), sha("binary"), sha("linkage")), anchor=anchor,
            evaluator=campaign.api.EvaluatorIdentity(
                id="autokernel.campaign-live-evaluation/v1",
                bundle_sha256=sha("evaluator"),
                runtime_source_label_ref="sha256:" + sha("source-label")),
            scope_denominator=campaign.api.ScopeDenominator(
                machine_subset="full", numa_nodes=(), devices=(), cores=192),
            scope_manifest_sha256=sha("scope"), co_residency="single",
            determinism=campaign.api.DeterminismReport("not_measured", 0),
            metric="prefill_tokens_per_s", metric_direction="higher_better",
            reps=5, change_class="parameter", anchor_tier="T0",
            transfer_ratio_to=(), created_at="2026-08-12T10:00:00+00:00",
            campaign_controls=controls, calibration=calibration)
        passed = schemas.Check(schemas.PASS)
        panel = campaign.api.ControlPanel(
            positive=passed, neutral=passed, degraded_negative=passed,
            aa=passed, historical_replay=passed)
        attestations = campaign.api.WindowAttestations(
            resource_claim_receipt="sha256:" + sha("claim"),
            resource_claim_open=passed, resource_claim_close=passed,
            resource_claim_same_holder=passed, no_concurrent_inference=passed,
            preflight_attestation_ref="sha256:" + sha("preflight"),
            host_receipt="sha256:" + sha("host"), host_health=passed,
            anchor_at_open=anchor, anchor_at_close=anchor, anchor_gate=passed,
            evaluator_bundle=passed, runtime_source_label=passed,
            recipe=campaign.api.RecipeReceipt(
                "recipe/v1", sha("constructor"), sha("argv")),
            storage_open=passed, storage_close=passed, strata=passed,
            stopping_rule_id="fixed-10/v1", rule_immutability=passed,
            order_randomized=passed, order_seed="ak-test/t0",
            aa_cadence=passed, controls=panel, calibration=passed,
            control_definitions_immutable=passed,
            raw_evidence_ref="sha256:" + sha("raw-t0"))
        for gate_id in (
                "t0.measurement_source_pin", "t0.build_evidence",
                "t0.compile_artifact_diff"):
            with self.subTest(gate_id=gate_id):
                ops = campaign.HostOps(nominal_khz=1)
                check = schemas.Check(schemas.FAIL, (f"induced {gate_id}",))
                outcome = ops._stop_t0_early(early_request, gate_id, check)
                self.assertFalse(outcome.all_pass)
                with mock.patch.object(
                        ops, "_window_attestations", return_value=attestations):
                    [event] = ops._evaluation_events(spec())
                self.assertEqual(schemas.validate_evaluation_event_v5(event), [])
                self.assertEqual(event["performance"]["raw_samples"], [])
                self.assertEqual(event["performance"]["paired_blocks"], 0)
                self.assertIsNone(event["performance"]["estimate"])
                self.assertIn(gate_id, event["stability"])

    def test_an_exception_before_event_construction_gets_a_distinct_t0_refusal(self):
        durable_parent = Path(campaign.__file__).resolve().parents[3] / "data"
        durable_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=durable_parent) as root:
            run_spec = spec(journal_root=root)
            ops = campaign.HostOps(nominal_khz=1)
            ops._t0_started = True
            result = mock.Mock(error="ValueError: identity could not be formed")
            ops.journal_evaluation(run_spec, result)
            entries = journal_module.Journal(
                root, campaign_id=run_spec.campaign_id).read_all()
            [refusal] = [entry for entry in entries
                         if entry.kind == journal_module.KIND_T0_REFUSAL]
            self.assertFalse(refusal.payload["rate_measured"])
            self.assertIn("identity could not be formed", refusal.payload["error"])

    def test_t0_and_t1_use_their_own_close_anchor_and_close_drift_voids(self):
        """The llama-cli T0 anchor must never be closed with llama-bench bytes."""
        from .evaluator import statistics

        passed = schemas.Check(schemas.PASS)
        sha = lambda value: __import__("hashlib").sha256(value.encode()).hexdigest()
        controls = campaign.api.CampaignControls(
            calibration_block_count=20, contribution_floor=0.02,
            max_candidates=10, confirmation_admission_count=2,
            max_blocks_per_candidate=40, storage_floor_bytes_free=1)
        calibration = campaign.api.CalibrationOutputs(
            backend="llama_cpu", phase="prefill", cell_class="tiny_real_graph",
            noise_floor_phi=0.009, b_min_blocks=10, alpha_sel=0.1,
            alpha_conf=0.05, anchor_gate_band=(0.97, 1.03), accepted=True,
            solve_order_recorded=campaign.api.CALIBRATION_SOLVE_ORDER,
            samples_ref="raw/calibration",
            e_process_construction_id="sign_martingale_predictable_lambda/v1")
        panel = campaign.api.ControlPanel(
            positive=passed, neutral=passed, degraded_negative=passed,
            aa=passed, historical_replay=passed)
        authority = control_runner.LiveEvaluationAuthority(
            campaign_controls=controls, calibration=calibration, controls=panel,
            aa_cadence=passed, control_definitions_immutable=passed,
            construction_id="ak.test-construction/v1",
            stopping_rule_id="ak.stopping.bounded_extension/v1", mde=0.021,
            runtime_source_label_ref="ake-srclabel-0003",
            evidence_ref="/durable/test-authority")
        lean = campaign.LeanCalibration(
            recipe_id=campaign.HISTORICAL_CALIBRATED_RECIPE_ID,
            contribution_floor=0.02, b_min_blocks=10, max_blocks=40,
            noise_floor_phi=0.009, mde=0.021,
            production_commit=campaign.PRODUCTION_COMMIT,
            measurement_commit=campaign.MEASUREMENT_COMMIT,
            evidence_ref=authority.evidence_ref, evaluation_authority=authority)
        run_spec = spec(blocks=12, calibration=lean)
        base_anchor = campaign.api.AnchorIdentity(
            source_commit=campaign.PRODUCTION_COMMIT,
            binary_sha256=sha("anchor-binary"),
            linkage_sha256=sha("anchor-linkage"))
        cli = base_anchor.for_tool("llama-cli")
        bench = base_anchor.for_tool("llama-bench")

        def request(tier, anchor, event_id):
            return campaign.api.EvaluationRequest(
                event_id=event_id, campaign_id="ak-test", candidate_id="akc-test",
                tier=tier, backend="llama_cpu", phase="prefill",
                cell_class="tiny_real_graph", protocol_id="P-AK-SEARCH-1/v1",
                artifact=campaign.api.ArtifactIdentity(
                    sha("source"), sha("binary"), sha("linkage")), anchor=anchor,
                evaluator=campaign.api.EvaluatorIdentity(
                    id="autokernel.campaign-live-evaluation/v1",
                    bundle_sha256=sha("evaluator"),
                    runtime_source_label_ref=authority.runtime_source_label_ref),
                scope_denominator=campaign.api.ScopeDenominator(
                    machine_subset="full", numa_nodes=(), devices=(), cores=192),
                scope_manifest_sha256=sha("scope"), co_residency="single",
                determinism=campaign.api.DeterminismReport("bitwise_stable", 3),
                metric="prefill_tokens_per_s", metric_direction="higher_better",
                reps=1, change_class="parameter", anchor_tier=tier,
                transfer_ratio_to=(), created_at="2026-08-12T10:00:00+00:00",
                campaign_controls=controls, calibration=calibration)

        t0_request = request("T0", cli, "ake-tool-t0")
        t1_request = request("T1", bench, "ake-tool-t1")
        gates = (
            campaign.api.GateResult("exact", campaign.api.GATE_CORRECTNESS,
                                    passed, requires_anchor=True),
            campaign.api.GateResult("stable", campaign.api.GATE_STABILITY, passed),
        )
        blocks = tuple(statistics.PairedBlock(
            block_index=index, unit_id=f"u{index}",
            stratum=campaign.api.STRATUM_SELECTION,
            order=(statistics.ORDER_ANCHOR_FIRST if index % 2 == 0 else
                   statistics.ORDER_CANDIDATE_FIRST),
            anchor_samples=(1.0,), candidate_samples=(1.061,))
            for index in range(12))
        rate_run = mock.Mock()
        rate_run.paired_blocks.return_value = blocks
        rate_run.order_control = passed
        rate_run.plan.campaign_seed = "campaign-seed-4711"

        ops = campaign.HostOps(nominal_khz=1)
        ops._claim_release_receipt = {"region": {"released_at": "now"}, "devices": []}
        ops._claim_open_check = ops._claim_close_check = passed
        ops._claim_same_holder_check = passed
        ops._preflight_check = ops._no_concurrent_close = passed
        ops._preflight_open_receipt = {"basis": "claim_witness", "when": "open"}
        ops._preflight_close_receipt = {"basis": "claim_witness", "when": "close"}
        ops._host_health_close = ops._storage_open = ops._storage_close = passed
        ops._evaluator_close_check = ops._runtime_close_check = passed
        ops._anchors_at_close = {"llama-cli": cli, "llama-bench": bench}
        ops._anchor_close_checks = {"llama-cli": passed, "llama-bench": passed}
        ops._t0_gate_results = gates
        retained_recipe = campaign.api.RecipeReceipt(
            "ak.test-recipe/v1", sha("constructor"), sha("argv"))
        ops._recipe_receipts = {
            ("T0", "llama-cli"): retained_recipe,
            ("T1", "llama-bench"): retained_recipe,
        }
        with mock.patch.object(
                ops, "_construct",
                side_effect=AssertionError("post-run recipe reconstruction")):
            t0_window = ops._window_attestations(
                run_spec, t0_request, raw_evidence_ref="raw/t0", rate_run=None)
            t1_window = ops._window_attestations(
                run_spec, t1_request, raw_evidence_ref="raw/t1", rate_run=rate_run)
        self.assertIs(t0_window.recipe, retained_recipe)
        self.assertIs(t1_window.recipe, retained_recipe)
        self.assertEqual(t0_window.anchor_at_close.tool, "llama-cli")
        self.assertEqual(t1_window.anchor_at_close.tool, "llama-bench")
        self.assertNotEqual(t0_window.preflight_attestation_ref,
                            "sha256:" + schemas.content_hash({
                                "outcome": passed.outcome, "reasons": []}))
        dispatcher = campaign.api.TierDispatcher(gate_runners={
            "T0": campaign._RecordedGateRunner(gates),
            "T1": campaign._RecordedGateRunner(gates),
        })
        self.assertNotEqual(
            dispatcher.dispatch(t0_request, t0_window).verdict.status,
            campaign.api.STATUS_INVALID)
        self.assertNotEqual(
            dispatcher.dispatch(t1_request, t1_window,
                                effect=campaign.api.EffectEstimate(
                                    metric="prefill_tokens_per_s",
                                    metric_direction="higher_better", value=0.061,
                                    e_value=20.0, threshold=10.0, mde=0.021,
                                    noise_floor=0.009, paired_blocks=12,
                                    stratum=campaign.api.STRATUM_SELECTION,
                                    raw_samples=tuple(0.061 for _ in range(12)),
                                    raw_samples_ref="raw/t1")).verdict.status,
            campaign.api.STATUS_INVALID)

        negative_windows = (
            ("anchor", mock.patch.dict(ops._anchors_at_close,
                                       {"llama-cli": bench})),
            ("evaluator", mock.patch.object(
                ops, "_evaluator_close_check",
                schemas.Check(schemas.FAIL, ("bundle drift",)))),
            ("claim", mock.patch.object(
                ops, "_claim_close_check",
                schemas.Check(schemas.FAIL, ("claim drift",)))),
            ("host", mock.patch.object(
                ops, "_host_health_close",
                schemas.Check(schemas.FAIL, ("host drift",)))),
        )
        for label, patcher in negative_windows:
            with self.subTest(drift=label), patcher:
                drifted = ops._window_attestations(
                    run_spec, t0_request, raw_evidence_ref="raw/t0", rate_run=None)
                outcome = dispatcher.dispatch(t0_request, drifted)
                self.assertEqual(outcome.verdict.status, campaign.api.STATUS_INVALID)


class TestVidyaBeliefCaptureProducer(unittest.TestCase):
    """The prospective write marker is identity- and raw-reduction-bound."""

    def event(self):
        from .test_journal import _event

        record = _event("belief-capture")
        raw = [
            [0, "u0", "selection", "anchor_first", "base", None,
             "2026-08-12T10:00:00+00:00", [100.0, 102.0], [104.0, 106.0]],
            [1, "u1", "selection", "candidate_first", "base", None,
             "2026-08-12T10:01:00+00:00", [99.0, 101.0], [103.0, 105.0]],
        ]
        effects = [
            (campaign.median(block[8]) - campaign.median(block[7])) /
            campaign.median(block[7]) for block in raw]
        record["performance"]["raw_samples"] = raw
        record["performance"]["raw_samples_ref"] = (
            "sha256:" + schemas.content_hash(raw))
        record["performance"]["paired_blocks"] = len(raw)
        record["performance"]["estimate"] = campaign.median(effects)
        record["claim_grammar"]["reps"] = 2
        record["performance"].setdefault("search_discipline", {})
        return record

    def test_capture_has_exact_schema_and_load_bearing_identity(self):
        record = self.event()
        produced = control_runner.attach_belief_capture(
            record, effect_scale="relative", model_id="model.gguf",
            model_sha256="a" * 64, producer_sha256="b" * 64)
        capture = produced["performance"]["search_discipline"]["belief_capture"]
        self.assertEqual(schemas.validate_evaluation_event_v5(produced), [])
        self.assertEqual(
            capture["schema"],
            "epyc.vidya.autokernel_evaluation_event_capture.v1")
        self.assertEqual(capture["raw_samples_sha256"],
                         schemas.content_hash(record["performance"]["raw_samples"]))
        original_binding = capture["identity_binding_sha256"]
        record["candidate_id"] = "akc-mutated"
        changed = control_runner.attach_belief_capture(
            record, effect_scale="relative", model_id="model.gguf",
            model_sha256="a" * 64, producer_sha256="b" * 64)
        self.assertNotEqual(
            original_binding,
            changed["performance"]["search_discipline"]["belief_capture"]
                   ["identity_binding_sha256"])

    @staticmethod
    def _root_vidya_contract_accepts(envelope):
        """Copied cross-contract from root commit 6c9cad04's SC10 adapter."""
        if (envelope.get("journal_schema") != "epyc.autokernel.journal_entry.v1"
                or envelope.get("kind") != "EVALUATION_EVENT"):
            return False
        event = envelope.get("payload")
        if (not isinstance(event, dict)
                or envelope.get("campaign_id") != event.get("campaign_id")
                or envelope.get("record_id") != event.get("event_id")):
            return False
        performance = event["performance"]
        capture = performance["search_discipline"].get("belief_capture")
        if not isinstance(capture, dict):
            return False
        raw = performance["raw_samples"]
        raw_sha = schemas.content_hash(raw)
        if (capture.get("raw_samples_sha256") != raw_sha
                or performance.get("raw_samples_ref") != f"sha256:{raw_sha}"
                or performance.get("paired_blocks") != len(raw)):
            return False
        claim = event["claim_grammar"]
        binding = {
            "schema": capture["schema"], "event_id": event["event_id"],
            "campaign_id": event["campaign_id"],
            "candidate_id": event["candidate_id"], "category": claim["category"],
            "protocol_id": claim["protocol_id"], "metric": claim["metric"],
            "metric_direction": claim["metric_direction"], "reps": claim["reps"],
            "effect_scale": capture["effect_scale"], "model_id": capture["model_id"],
            "model_sha256": capture["model_sha256"],
            "source_sha256": capture["source_sha256"],
            "binary_sha256": capture["binary_sha256"],
            "resource_claim_receipt": capture["resource_claim_receipt"],
            "producer_sha256": capture["producer_sha256"],
            "raw_samples_sha256": capture["raw_samples_sha256"],
        }
        if capture.get("identity_binding_sha256") != schemas.content_hash(binding):
            return False
        effects = []
        for block in raw:
            if (not isinstance(block, list) or len(block) != 9
                    or len(block[7]) != claim["reps"]
                    or len(block[8]) != claim["reps"]):
                return False
            anchor = campaign.median(block[7])
            candidate = campaign.median(block[8])
            effects.append((candidate - anchor) / anchor)
        return math.isclose(
            campaign.median(effects), performance["estimate"],
            rel_tol=1e-12, abs_tol=1e-15)

    def test_actual_journal_envelope_passes_root_vidya_sc10_contract(self):
        record = control_runner.attach_belief_capture(
            self.event(), effect_scale="relative", model_id="model.gguf",
            model_sha256="a" * 64, producer_sha256="b" * 64)
        durable_parent = Path(campaign.__file__).resolve().parents[3] / "data"
        durable_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=durable_parent) as root:
            book = journal_module.Journal(
                root, campaign_id=record["campaign_id"])
            book.initialize()
            entry = book.append(journal_module.KIND_EVALUATION_EVENT, record)
            self.assertTrue(self._root_vidya_contract_accepts(entry.envelope()))

    def test_capture_refuses_reps_or_raw_hash_drift(self):
        record = self.event()
        record["claim_grammar"]["reps"] += 1
        with self.assertRaisesRegex(ValueError, "reps"):
            control_runner.attach_belief_capture(
                record, effect_scale="relative", model_id="model.gguf",
                model_sha256="a" * 64, producer_sha256="b" * 64)
        record = self.event()
        record["performance"]["raw_samples"][0][8][0] += 1.0
        with self.assertRaisesRegex(ValueError, "raw sample hash"):
            control_runner.attach_belief_capture(
                record, effect_scale="relative", model_id="model.gguf",
                model_sha256="a" * 64, producer_sha256="b" * 64)
