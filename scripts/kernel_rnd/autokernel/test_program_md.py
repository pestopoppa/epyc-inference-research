"""`program.md` is a control surface, so its claims are executed, not proofread.

The file tells a session which command to type, which symbols to call, what the
accept rule is, and what a terminal state means. Every one of those is a claim
about code, and an unchecked claim about code is how a runbook goes stale
without anybody noticing. Each test below pins something that was WRONG in the
file on 2026-08-04, found by running it rather than reading it:

* The documented invocation was `PYTHONPATH=scripts/kernel_rnd python3 -m
  autokernel.campaign`, which `build_parser()`'s own `prog` contradicts — in a
  file that declares `--help` authoritative over itself.
* The accept rule was described as `api.compute_verdict`. The driver does not
  call it: it applies `campaign.decide`. Two rules, one of them documented, the
  other one executed.
* The worked example printed `4 COULD_NOT_CHECK` beside `status: pass` and
  `speed_rank_admissible: True`. `compute_verdict` cannot return that —
  COULD_NOT_CHECK demotes every gate class to `inconclusive`. A reader
  calibrating "healthy" against an impossible block either believes the loop is
  broken or softens `_ON_GATE_COULD_NOT_CHECK` to make the example true, which
  is tuning the instrument.
* The stop conditions and four loop states were `controller/` vocabulary, so the
  runbook routed the loop through the half the operator has not adopted. The
  IMPORT boundary is owned by `test_campaign_footprint.py`, which walks the real
  graph including lazy, conditional and `importlib` edges — this file only keeps
  the PROSE from sending a reader somewhere the imports may not go.

Every guard here has a compliant-path control, because a boundary test that a
session could satisfy by emptying the file is not a boundary test.
"""

from __future__ import annotations

import importlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[1])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import campaign  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel.evaluator import api  # noqa: E402
from autokernel.execution import physical_bounds  # noqa: E402
from autokernel.test_schemas import _proposal as _proposal_fixture  # noqa: E402

PROGRAM_MD = Path(__file__).resolve().parent / "program.md"


def _write_current_calibration(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    source = {
        "schema": "epyc.autokernel.runtime_source_label.v1",
        "production_source_commit": campaign.PRODUCTION_COMMIT,
        "measurement_instrument_commit": campaign.MEASUREMENT_COMMIT,
        "measurement_binary_sha256": "1" * 64,
        "copied_binary_sha256": "1" * 64,
        "measurement_linkage_sha256": "2" * 64,
        "copied_linkage_sha256": "2" * 64,
        "binary_copy_exact": True,
    }
    source_sha = S.content_hash(source)
    (root / "runtime-source-label.json").write_text(
        json.dumps({**source, "source_sha256": source_sha}), encoding="utf-8")
    declaration = {
        "schema": "epyc.autokernel.live_control_campaign_declaration.v1",
        "campaign_id": "ak-controls-current-doc-test",
        "recipe_id": campaign.HISTORICAL_CALIBRATED_RECIPE_ID,
        "contribution_floor": 0.03,
        "max_blocks_per_candidate": 20,
        "source_sha256": source_sha,
    }
    (root / "campaign_declaration.json").write_text(
        json.dumps(declaration), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({
        "campaign_id": declaration["campaign_id"],
        "state": "controls_complete", "may_rank": True,
        "binary_copy_exact": True,
        "production_source_commit": campaign.PRODUCTION_COMMIT,
        "calibration": {
            "outputs": {"accepted": True, "b_min_blocks": 12,
                        "noise_floor_phi": 0.04},
            "attempts": [{"accepted": True,
                          "mde": {"found": True, "value": 0.025}}],
        },
    }), encoding="utf-8")

#: Subpackages the operator has not adopted into campaign #1 and which are STILL
#: ON DISK. `release/` and `adapters/` were restored narrowly for the AK9 speech
#: release compiler on 2026-08-12; restoration does not make them part of the
#: mutation/build path, so `program.md` still may not route a campaign into them.
DEFERRED_PACKAGES = ("controller", "release", "adapters")

#: Deleted on the operator's approval, 2026-08-04 (tag
#: `autokernel-preserve-20260804`) and not subsequently restored. `program.md`
#: may not name these either, and the reason is a different one worth keeping
#: separate: not that a reader would be routed into the unadopted half, but that
#: a runbook citing a module which no longer exists is the same staleness
#: arriving by another route.
#:
#: They are a separate tuple because the CONTROL below is inverted for them —
#: "still on disk" is exactly what must NOT hold. Left in the deferred list, they
#: made that control pass over an empty directory containing nothing but a
#: `__pycache__`, which is the vacuity it was written to prevent.
DELETED_PACKAGES = ("surface",)

#: Modules of the adopted half, by the leaf name `program.md` cites them under.
ESSENTIAL_MODULES = {
    "campaign": "autokernel.campaign",
    "worktree": "autokernel.execution.worktree",
    "microbench": "autokernel.execution.microbench",
    "t0_provider": "autokernel.execution.t0_provider",
    "cpu_region_claim": "autokernel.execution.cpu_region_claim",
    "chain": "autokernel.execution.chain",
    "device_claim": "autokernel.resource.device_claim",
    "claim_witness": "autokernel.resource.claim_witness",
    "preflight": "autokernel.resource.preflight",
    "api": "autokernel.evaluator.api",
    "recipes": "autokernel.evaluator.recipes",
    "correctness": "autokernel.evaluator.correctness",
    "devices": "autokernel.evaluator.devices",
    "statistics": "autokernel.evaluator.statistics",
    "journal": "autokernel.journal",
    "storage": "autokernel.storage",
    "schemas": "autokernel.schemas",
}

#: `head.attr` inside backticks, optionally package-qualified. `.py` is a file
#: extension, not an attribute.
_CITATION = re.compile(
    r"`(?:[a-z_]+\.)*(" + "|".join(ESSENTIAL_MODULES) + r")\.([A-Za-z_][A-Za-z0-9_]*)")


def text() -> str:
    return PROGRAM_MD.read_text(encoding="utf-8")


class TestTheProseStaysInsideTheAdoptedHalf(unittest.TestCase):
    """`test_campaign_footprint.py` owns the import graph; this owns the prose.

    They are not the same guarantee. `program.md` is what the next session reads
    before it writes anything, so a runbook that hands the loop's stop authority
    to `controller/` defeats the import boundary one session later, from the
    outside.
    """

    def test_no_deferred_package_is_named(self):
        body = text()
        hits = []
        for name in DEFERRED_PACKAGES + DELETED_PACKAGES:
            # `release/` or `release.symbol` — not the English word "release."
            for match in re.finditer(rf"\b{name}(?:/|\.[A-Za-z_])", body):
                hits.append(f"{name!r} at line {body.count(chr(10), 0, match.start()) + 1}")
        self.assertEqual(hits, [], "program.md routes a reader into the deferred half")

    def test_the_deferred_packages_are_still_on_disk(self):
        """The control: this guard means nothing if the deferred half is gone.

        Deleting it is the operator's call and stays a separate one; when it is
        made, this test says so instead of passing silently — which is what it did
        on 2026-08-04. `release/` and `adapters/` had been deleted then, but the
        directories survived holding a stale `__pycache__`, so `is_dir()` was
        still true and the control reported a boundary over files that no longer
        existed. They are now deliberately restored and classified as deferred;
        both directions remain asserted, so neither an unrecorded deletion nor a
        leftover shell can pass.
        """
        package = Path(__file__).resolve().parent
        for name in DEFERRED_PACKAGES:
            self.assertTrue((package / name).is_dir(),
                            f"{name}/ is gone — the boundary this file guards no longer "
                            "describes the tree, and this test list is stale")
        for name in DELETED_PACKAGES:
            self.assertFalse((package / name).exists(),
                             f"{name}/ is listed as deleted but is back on disk; it is "
                             "either a real restoration — in which case it belongs in "
                             "DEFERRED_PACKAGES — or a leftover shell that makes the "
                             "check above pass over nothing")

    def test_the_adopted_half_is_still_named(self):
        """The compliant-path control: the guard above must not be passable by
        deleting the file's content."""
        body = text()
        for leaf in ("campaign", "worktree", "t0_provider", "correctness",
                     "recipes", "journal", "storage", "schemas"):
            self.assertIn(leaf, body,
                          f"program.md no longer names {leaf}, which is adopted and "
                          "load-bearing — the boundary test would pass on an empty file")


class TestEverySymbolProgramMdCitesResolves(unittest.TestCase):
    """A runbook that names a symbol nobody can call is worse than silence."""

    def test_citations_resolve(self):
        body = text()
        missing = []
        for leaf, attr in dict.fromkeys(
                (m.group(1), m.group(2)) for m in _CITATION.finditer(body)):
            if attr == "py":
                continue
            module = importlib.import_module(ESSENTIAL_MODULES[leaf])
            if not hasattr(module, attr):
                missing.append(f"{leaf}.{attr} is not in {module.__name__}")
        self.assertEqual(missing, [], "program.md cites symbols that do not exist")

    def test_the_scan_actually_found_citations(self):
        """The control: an empty scan passes the test above vacuously."""
        found = {(m.group(1), m.group(2)) for m in _CITATION.finditer(text())}
        self.assertGreaterEqual(len(found), 12, sorted(found))


class TestTheDocumentedCommandsAreTheRealCommands(unittest.TestCase):
    """`program.md` declares `--help` authoritative over itself. Hold it to that.

    It documented `PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.campaign`
    while the parser's own `prog` is `python3 -m
    scripts.kernel_rnd.autokernel.campaign`. A session that types the file's
    version gets an import error before it gets a campaign.
    """

    def _documented_commands(self):
        block = re.search(r"```bash\n(.*?)```", text(), re.S)
        self.assertIsNotNone(block, "the Setup command block is gone from program.md")
        joined = block.group(1).replace("\\\n", " ")
        # Any line that invokes a module, however it is prefixed. Selecting on a
        # `python3` prefix would let the exact bug this test exists for slip
        # through: the wrong invocation was spelled
        # `PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.campaign`, and a
        # startswith filter drops it instead of failing on it.
        commands = [line.strip() for line in joined.splitlines() if " -m " in line]
        self.assertTrue(commands, "no module invocation is documented in Setup")
        return commands

    def test_the_documented_module_path_is_the_parsers_own_prog(self):
        prog = campaign.build_parser().prog
        for command in self._documented_commands():
            self.assertIn(prog, command,
                          f"program.md documents {command!r}, which is not {prog!r} — the "
                          "file declares --help authoritative over itself")

    def test_every_documented_command_actually_runs(self):
        """Parsing is not running, and the gap is where the bug was.

        `--campaign-id ak-aug05 --candidate-id akc-0001` PARSES and then exits 2
        on `required parameter 'model' is missing`, because `--model` has no
        default. A session typing the documented line got a refusal, not a
        campaign. This drives `main()` in dry-run, which acquires nothing,
        spawns nothing and builds nothing.
        """
        checked = 0
        for command in self._documented_commands():
            parts = command.split()
            argv = parts[parts.index("scripts.kernel_rnd.autokernel.campaign") + 1:]
            if "--help" in argv:
                continue
            argv = [a for a in argv if a not in ("--execute", "--i-hold-the-host")]
            tempdir = tempfile.TemporaryDirectory()
            self.addCleanup(tempdir.cleanup)
            if "--proposal-manifest" in argv:
                proposal = _proposal_fixture()
                # The shared schema fixture intentionally predates the live
                # instrument.  The documented command must still exercise the
                # current campaign contract rather than fail on that fixture's
                # historical source pin.
                proposal["provider_reference"]["source_commit"] = (
                    campaign.MEASUREMENT_COMMIT)
                campaign_id = argv[argv.index("--campaign-id") + 1]
                proposal["campaign_id"] = campaign_id
                proposal["proposal_id"] = "akp-documented-command"
                proposal["provider_reference"]["target_backend"] = (
                    argv[argv.index("--backend") + 1]
                    if "--backend" in argv else campaign.BACKEND_CPU)
                manifest = Path(tempdir.name) / "proposal.json"
                manifest.write_text(json.dumps(proposal), encoding="utf-8")
                argv[argv.index("--proposal-manifest") + 1] = str(manifest)
            if "--calibration-bundle" in argv:
                bundle = Path(tempdir.name) / "calibration"
                _write_current_calibration(bundle)
                argv[argv.index("--calibration-bundle") + 1] = str(bundle)
            if "--physical-envelope" in argv:
                calibration = campaign.load_calibration_bundle(
                    argv[argv.index("--calibration-bundle") + 1])
                model = argv[argv.index("--model") + 1]
                built = campaign.CampaignSpec(
                    campaign_id=argv[argv.index("--campaign-id") + 1],
                    candidate_id=argv[argv.index("--candidate-id") + 1],
                    candidate_ref=argv[argv.index("--candidate") + 1],
                    recipe_id=calibration.recipe_id,
                    blocks=calibration.b_min_blocks,
                    model=model,
                    calibration=calibration,
                )
                envelope = physical_bounds.PhysicalEnvelope(
                    shape_id=built.measurement_unit_id,
                    delivered_unit="token", flops_per_unit=1.0,
                    bytes_per_unit=1.0, peak_compute_flops_s=1e15,
                    peak_memory_bytes_s=1e15,
                    measurement_frame_sha256=physical_bounds.measurement_frame_sha256(
                        built.recipe_id, built.bench_params),
                    work_derivation_ref="documented-command test fixture",
                    hardware_peak_ref="documented-command test fixture",
                )
                envelope_path = Path(tempdir.name) / "physical-envelope.json"
                envelope_path.write_text(
                    json.dumps(envelope.to_dict()), encoding="utf-8")
                argv[argv.index("--physical-envelope") + 1] = str(envelope_path)
            out, err = io.StringIO(), io.StringIO()
            stderr, sys.stderr = sys.stderr, err
            try:
                code = campaign.main(argv, out=out)
            finally:
                sys.stderr = stderr
            self.assertEqual(code, 0,
                             f"the documented command refuses to start: {err.getvalue()}")
            self.assertIn(campaign.STATE_COMPOSED, out.getvalue(),
                          "a documented command must reach a dry-run composition")
            checked += 1
        self.assertGreater(checked, 0, "no runnable command is documented in Setup")

    def test_dry_run_is_the_parser_default_not_a_flag_you_remember(self):
        self.assertTrue(campaign.build_parser().parse_args([]).dry_run,
                        "a flagless invocation must not be the thing that starts a benchmark")

    def test_execute_without_the_host_attestation_refuses(self):
        stderr, sys.stderr = sys.stderr, io.StringIO()
        try:
            code = campaign.main(["--execute"])
        finally:
            sys.stderr = stderr
        self.assertEqual(code, 2, "--execute must refuse without --i-hold-the-host")

    def test_program_md_says_both_flags_are_needed(self):
        body = text()
        self.assertIn("--i-hold-the-host", body,
                      "program.md must document the attestation --execute requires, or a "
                      "session hits exit 2 with no idea why")


class TestTheAcceptRuleProgramMdDescribesIsTheOneThatRuns(unittest.TestCase):
    """The three attacks the A/A was measured to answer, run against `decide`."""

    T0_OK = None
    ORDERS = ("anchor_first", "candidate_first")

    def setUp(self):
        self.t0 = campaign.T0Outcome(all_pass=True,
                                     gates=(("mul_mat_id", "PASS", ("ok",)),))

    def _pairs(self, anchors, candidates):
        return [campaign.Pair(block_index=i, anchor=a, candidate=c,
                              order=self.ORDERS[i % 2])
                for i, (a, c) in enumerate(zip(anchors, candidates))]

    def test_the_measured_aa_null_is_refused_in_every_alignment(self):
        """The real null, with the real drift: four A/A runs of IDENTICAL code
        used as both arms. Any alignment that accepts is a rule that accepts
        nothing happening."""
        import itertools
        for metric, series in (("decode_tokens_per_s", campaign.AA_TG128_RUNS),
                               ("prefill_tokens_per_s", campaign.AA_PP512_RUNS)):
            bound = campaign.drift_bound_for_metric(metric)
            runs = list(series)
            for perm in itertools.permutations(range(len(runs))):
                decision = campaign.decide(
                    self._pairs(runs, [runs[i] for i in perm]),
                    t0=self.t0, blocks_precommitted=len(runs), drift_bound=bound)
                self.assertFalse(decision.keep,
                                 f"{metric}: the A/A null was accepted under {perm}")

    def test_a_real_win_is_still_accepted(self):
        """The compliant-path control: a rule that rejects the null by rejecting
        everything is not a rule."""
        decision = campaign.decide(
            self._pairs([50.0] * 5, [53.0] * 5), t0=self.t0, blocks_precommitted=5,
            drift_bound=campaign.drift_bound_for_metric("decode_tokens_per_s"))
        self.assertTrue(decision.keep, decision.reason)

    def test_a_fast_wrong_kernel_gets_no_speed_number_at_all(self):
        failed = campaign.T0Outcome(
            all_pass=False, gates=(("mul_mat_id", "FAIL", ("MoE dispatch mismatch",)),))
        with self.assertRaises(campaign.AcceptRuleMisuse):
            campaign.decide(self._pairs([50.0] * 5, [99.0] * 5), t0=failed,
                            blocks_precommitted=5, drift_bound=0.0213)

    def test_n_cannot_be_extended_after_seeing_the_result(self):
        with self.assertRaises(campaign.AcceptRuleMisuse):
            campaign.decide(self._pairs([50.0] * 9, [55.0] * 9), t0=self.t0,
                            blocks_precommitted=5, drift_bound=0.0213)

    def test_program_md_names_the_rule_the_driver_applies(self):
        body = text()
        self.assertIn("campaign.decide", body,
                      "program.md must name the accept rule the driver actually applies; it "
                      "described api.compute_verdict, which the driver never calls")
        self.assertIn("min(delta) > 0", body)
        self.assertIn("drift_bound", body)

    def test_the_drift_bound_is_derived_from_the_recorded_aa_not_typed(self):
        """Nobody loosens the bound without changing a measurement."""
        self.assertAlmostEqual(
            campaign.DRIFT_BOUND_BY_METRIC["decode_tokens_per_s"],
            campaign.drift_bound_from(campaign.AA_TG128_RUNS))
        self.assertGreater(campaign.DRIFT_BOUND_BY_METRIC["prefill_tokens_per_s"], 0.0)


class TestTheSecondVerdictLayerIsDescribedHonestly(unittest.TestCase):
    """`program.md` now states two facts about `api.Verdict`. Both are executed.

    They matter because a future `campaign.py` may be wired to the verdict layer,
    and the two traps below are what it would walk into.
    """

    @staticmethod
    def _verdict(gates, *, value: float) -> api.Verdict:
        return api.compute_verdict(
            tier="T1", gates=tuple(gates),
            void_scan=api.VoidScan(findings=(), evaluated=api.VOID_REASONS,
                                   not_applicable=()),
            search_grade=api.SearchGradeResult(
                satisfied=True, evaluated=tuple(c.id for c in api.SEARCH_GRADE_CONJUNCTS),
                failed=(), not_applicable=(), reasons=()),
            anchor=api.AnchorIdentity(
                source_commit="67a433bf45a8a091d83b4ea0b32ff0735fd51800",
                binary_sha256="0" * 64, linkage_sha256="1" * 64,
                measurement_event_ids=("ake-anchor-0001",)),
            effect=api.EffectEstimate(
                metric="decode_tokens_per_s", metric_direction="higher_better", value=value,
                e_value=5000.0, threshold=100.0, mde=0.024, noise_floor=0.019,
                paired_blocks=5, stratum=api.STRATUM_SELECTION,
                raw_samples=((100.0, 100.0 * (1.0 + value)),), raw_samples_ref="ak-raw://x"))

    def test_a_structurally_unproduced_surface_makes_pass_unreachable(self):
        gates = [api.GateResult(gate_id=f"p{i}", gate_class=api.GATE_CORRECTNESS,
                                check=S.Check(S.PASS, ("produced",))) for i in range(13)]
        gates += [api.GateResult(gate_id=g, gate_class=api.GATE_CORRECTNESS,
                                 check=S.Check(S.COULD_NOT_CHECK, ("no producer",)))
                  for g in ("sanitizer.asan", "sanitizer.ubsan",
                            "state_rollback_teardown_race", "exact_reference_comparison")]
        verdict = self._verdict(gates, value=0.029)
        self.assertEqual(verdict.status, "inconclusive")
        self.assertFalse(verdict.speed_rank_admissible)

    def test_the_same_gates_with_producers_do_reach_pass(self):
        """The control: `pass` is unreachable because of the four, not in general."""
        gates = [api.GateResult(gate_id=f"p{i}", gate_class=api.GATE_CORRECTNESS,
                                check=S.Check(S.PASS, ("produced",))) for i in range(13)]
        verdict = self._verdict(gates, value=0.029)
        self.assertEqual(verdict.status, "pass")
        self.assertTrue(verdict.speed_rank_admissible)

    def test_a_correctness_failure_and_a_parity_resolution_coexist(self):
        """Reading `effect_resolution` without `status` accepts a broken kernel."""
        gates = (
            api.GateResult(gate_id="produced", gate_class=api.GATE_CORRECTNESS,
                           check=S.Check(S.PASS, ("ok",))),
            api.GateResult(gate_id="mul_mat_id", gate_class=api.GATE_CORRECTNESS,
                           check=S.Check(S.FAIL, ("MUL_MAT_ID mismatch",))),
        )
        verdict = self._verdict(gates, value=0.002)
        self.assertEqual(verdict.status, "fail")
        self.assertTrue(api.is_sub_floor_resolution(verdict.effect_resolution))
        self.assertFalse(verdict.speed_rank_admissible)

    def test_program_md_warns_about_both(self):
        body = text()
        self.assertIn("speed_rank_admissible", body)
        self.assertRegex(body, r"status=fail.*below_noise_floor|below_noise_floor.*status=fail")


class TestTheStateVocabularyIsComplete(unittest.TestCase):
    """The terminal states program.md documents are the ones the driver emits."""

    def _documented(self) -> set:
        return set(re.findall(r"^state: (\w+)", text(), re.M))

    def test_every_documented_state_is_real(self):
        real = {getattr(campaign, n) for n in dir(campaign) if n.startswith("STATE_")}
        self.assertTrue(self._documented() <= real,
                        f"program.md documents states the driver cannot emit: "
                        f"{sorted(self._documented() - real)}")

    def test_no_real_state_is_undocumented(self):
        """The control: passing by documenting only one state is not passing.

        A session that meets an undocumented terminal state has to guess.
        """
        real = {getattr(campaign, n) for n in dir(campaign) if n.startswith("STATE_")}
        self.assertEqual(real - self._documented(), set(),
                         "program.md must document every terminal state the driver emits")


if __name__ == "__main__":
    unittest.main()
