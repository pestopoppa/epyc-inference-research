"""The screen->confirm keep gate at the LOOP boundary (§5.3, D1-D6).

R23-5 is the incident: +17.26% on the screen rung at ne11=1 was a decisive -1.46%
REGRESSION at ne11=8 on the production shape. These tests pin the three §5.3
contracts: a confirmed candidate is promoted to `kept`; a confirm regression lands
as `keep_candidate` -- never `kept` -- with BOTH measurements recorded; and with no
confirm configured the behavior is exactly the single-rung keep of today.
"""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from autokernel.controller import rung_confirm
from autokernel.loop import bench, loop, production, status
from autokernel.loop.test_loop import _Critic, _Planner, _comparison, _run


def _confirm_row(surface, effect, *, decisive):
    """A confirm-rung measurement double: exactly the fields the gate reads."""
    return types.SimpleNamespace(
        surface=surface, effect=effect, decisive=decisive, noise_floor_pct=0.7,
        to_dict=lambda: {"surface": surface, "effect": effect,
                         "decisive": decisive, "model": "Qwen3.8-27B-Q8_0.gguf"})


def _gate(store, floors=None):
    """A configured Confirm with fixture parity records -- no GGUF reads."""
    parity = mock.Mock()
    parity.to_dict = lambda: {"exact": True, "waived": False}
    waived = mock.Mock()
    waived.to_dict = lambda: {"exact": False, "waived": True}
    return rung_confirm.Confirm(
        model=Path("/models/Qwen3.8-27B-Q8_0.gguf"), pairs=5,
        surfaces=("dec-b4", "dec-b8"), store=Path(store),
        floors=floors if floors is not None else {"dec-b4": 0.751, "dec-b8": 0.9},
        screen_parity=waived, confirm_parity=parity)


class AConfirmedCandidateIsPromotedToKept(unittest.TestCase):

    def test_the_commit_gates_then_commits_and_iterate_reports_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _gate(tmp)
            rows = {"dec-b4": _confirm_row("dec-b4", +0.012, decisive=True),
                    "dec-b8": _confirm_row("dec-b8", +0.002, decisive=False)}
            committed = {}

            def commit(hypothesis, paths, comparison):
                verdict = gate.gate(hypothesis.mechanism_id, comparison,
                                    lambda surface, floor: rows[surface])
                if not verdict["promoted"]:
                    raise loop.ConfirmVetoed(verdict["reason"])
                committed["head"] = "abc1234"
                return "abc1234"

            outcome = loop.iterate(
                planner=_Planner(), critic=_Critic([], []), context={},
                measure=lambda h, p: _comparison(0.05), gate=lambda h, p: (True, []),
                commit=commit)
            self.assertEqual(outcome.status, "kept")
            self.assertEqual(committed.get("head"), "abc1234")
            record = self._only_record(Path(tmp))
            self.assertTrue(record["promoted"])
            # A non-decisive confirm surface is NOT a veto: the screen proved the
            # positive; the confirm's job is catching production-shape regressions.
            self.assertIn("confirmed", record["reason"])

    def _only_record(self, store: Path) -> dict:
        files = list((store / "confirm").glob("*.json"))
        self.assertEqual(len(files), 1, files)
        return json.loads(files[0].read_text(encoding="utf-8"))


class AConfirmRegressionIsNeverKept(unittest.TestCase):

    def _veto_run(self, tmp, rows):
        gate = _gate(tmp)
        committed = {}

        def commit(hypothesis, paths, comparison):
            verdict = gate.gate(hypothesis.mechanism_id, comparison,
                                lambda surface, floor: rows[surface])
            if not verdict["promoted"]:
                raise loop.ConfirmVetoed(verdict["reason"])
            committed["head"] = "abc1234"          # pragma: no cover - must not run
            return "abc1234"

        outcome = loop.iterate(
            planner=_Planner(), critic=_Critic([], []), context={},
            measure=lambda h, p: _comparison(0.05), gate=lambda h, p: (True, []),
            commit=commit)
        return outcome, committed

    def test_the_r23_5_inversion_lands_as_keep_candidate_with_both_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = {"dec-b4": _confirm_row("dec-b4", +0.005, decisive=True),
                    "dec-b8": _confirm_row("dec-b8", -0.0146, decisive=True)}
            outcome, committed = self._veto_run(tmp, rows)
            self.assertEqual(outcome.status, "keep_candidate",
                             "a vetoed screen keep is a candidate, never kept")
            self.assertNotIn("head", committed,
                             "the champion must NOT advance past a confirm veto")
            self.assertIn("dec-b8", outcome.reasons[0])
            self.assertIn("regression", outcome.reasons[0])
            self.assertIsNotNone(outcome.comparison,
                                 "the screen measurement stays on the outcome")
            record = json.loads(next(
                (Path(tmp) / "confirm").glob("*.json")).read_text(encoding="utf-8"))
            self.assertFalse(record["promoted"])
            self.assertEqual(record["screen"]["surface"], "tg128",
                             "the record carries the SCREEN measurement")
            self.assertEqual([row["surface"] for row in record["confirm"]],
                             ["dec-b4", "dec-b8"],
                             "...and every CONFIRM measurement beside it")
            self.assertTrue(record["parity"]["screen"]["waived"],
                            "the §5.1 waiver is a visible artifact on the record")

    def test_an_uncalibrated_confirm_surface_vetoes_fail_closed(self):
        """The same keep-refusal doctrine as the screen: no floor, no promotion --
        promoting through an unmeasured floor is the fake-decisive defect with a
        second rung."""
        with tempfile.TemporaryDirectory() as tmp:
            rows = {"dec-b4": _confirm_row("dec-b4", +0.02, decisive=None),
                    "dec-b8": _confirm_row("dec-b8", +0.01, decisive=True)}
            outcome, committed = self._veto_run(tmp, rows)
            self.assertEqual(outcome.status, "keep_candidate")
            self.assertNotIn("head", committed)
            self.assertIn("UNCALIBRATED", outcome.reasons[0])


class UnconfiguredMeansSingleRungExactlyAsToday(unittest.TestCase):
    """With no confirm model the keep path must be bit-identical to the old one."""

    def test_a_decisive_positive_still_commits_directly(self):
        outcome, committed = _run(_Planner(), _Critic([], []), effect=0.05)
        self.assertEqual(outcome.status, "kept")
        self.assertEqual(committed.get("head"), "abc1234")

    def test_the_flags_default_off_and_the_gate_is_guarded(self):
        """Wiring seams, in the style of test_gates: the defaults keep the running
        run's semantics untouched until the boundary enables the confirm rung."""
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        self.assertIn('parser.add_argument("--confirm-model", type=Path, '
                      'default=None', source)
        block = source.split("def commit_pooled(", 1)[1][:2200]
        self.assertIn("if confirm is not None:", block,
                      "unconfigured, commit_pooled must skip the gate entirely")
        self.assertIn("refuse_uncalibrated_keep", block.split(
            "if confirm is not None:", 1)[0],
            "the screen's own refusal still runs FIRST, gate or no gate")

    def test_confirm_defaults_match_the_operator_decisions(self):
        self.assertEqual(rung_confirm.DEFAULT_SURFACES, ("dec-b4", "dec-b8"),
                         "D2: dec-b4 + dec-b8 catch the R23-5 inversion class")
        self.assertEqual(rung_confirm.DEFAULT_PAIRS, 5,
                         "D3: the calibrated k=5 floor row")


class TheRungIsOnEveryRecord(unittest.TestCase):
    """§5.3: with two rungs live, a record without its model is a number without
    its instrument, and run histories across rungs must never merge."""

    def test_a_comparison_records_the_model_it_measured(self):
        with mock.patch.object(bench, "run_once",
                               lambda *a, **k: (100.0, {"resident": True,
                                                        "peak_vram_bytes": 1 << 31,
                                                        "peak_kfd_processes": 1})):
            comparison = bench.compare(
                bench.Arm("a", Path("/x")), bench.Arm("b", Path("/y")),
                Path("/models/Qwen3.8-27B-Q8_0.gguf"), pp=0, tg=128, pairs=1,
                warmup_pairs=0)
        self.assertEqual(comparison.model, "/models/Qwen3.8-27B-Q8_0.gguf")
        self.assertEqual(comparison.to_dict()["model"],
                         "/models/Qwen3.8-27B-Q8_0.gguf")

    def test_the_status_surface_carries_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e", campaign_id="c",
                         anchor_commit="a", surface="dec-b4", pairs=5,
                         noise_floor_pct=0.751, model="m.gguf")
            body = status.read(Path(tmp))
        self.assertEqual(body["model"], "m.gguf")

    def test_a_keep_candidate_counts_as_a_measurement_reached(self):
        with tempfile.TemporaryDirectory() as tmp:
            status.write(Path(tmp), state="running", epoch="e", campaign_id="c",
                         anchor_commit="a", surface="dec-b4", pairs=5,
                         noise_floor_pct=0.751,
                         outcomes=[{"status": "keep_candidate"},
                                   {"status": "measured_null"}])
            body = status.read(Path(tmp))
        self.assertEqual(body["measurements_reached"], 2,
                         "a vetoed candidate WAS measured -- twice, in fact")

    def test_the_headline_bundle_records_its_rung(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            base = store / "base"
            (base / "bin").mkdir(parents=True)
            (base / "bin" / "llama-bench").write_text("", encoding="utf-8")
            result = production.refresh(
                store=store, champion_commit="c" * 40,
                champion_build=store / "champ", baseline_build=base,
                resolve=lambda: ("f" * 40, "production-consolidated-v9"),
                compare=lambda _b, _c: bench.Comparison(
                    surface="dec-b4", anchor_samples=[100.0],
                    candidate_samples=[102.0], effect=0.02,
                    estimator="median_over_median", pairs=20,
                    noise_floor_pct=0.668, residency={},
                    model="Qwen3.8-27B-Q8_0.gguf"))
            self.assertTrue(result.published, result.reason)
            bundle = json.loads(
                (store / production.FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(bundle["model"], "Qwen3.8-27B-Q8_0.gguf")


if __name__ == "__main__":
    unittest.main()
