"""`P-AK-SEARCH-1-A3` at the loop boundary: what ranking may touch, and what it may not.

The store's own ranking is tested in `controller/test_experiments.py`. This file tests
the three places the loop could still get it wrong:

  * `archive.recall` must default to the pre-A3 behaviour and pass the flag through;
  * `run.py` must expose the opt-in, defaulted OFF, so an ordering that influenced a
    run is attributable to a flag someone typed;
  * NOTHING on the keep/null path may read a prior record's magnitude. A3 narrows
    denial 4 to permit ORDERING and nothing else, and the campaign calibration block is
    explicitly untouched: a campaign still derives its own thresholds. A stale effect
    size reaching the arithmetic behind `comparison.decisive and comparison.effect > 0`
    would be exactly the thing denial 4 was written to stop.

The last one is asserted as an impossibility rather than an observation. The rows
handed to the loop are booby-trapped: reading a magnitude out of them raises. A test
that simply does not perform the forbidden read would pass just as happily against
code that does.
"""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.controller import experiments as ex
from autokernel.controller import experiments_fixture as fx
from autokernel.loop import actors, archive, bench, gates, loop, run


class _Stop(Exception):
    """Abort `run.main` the instant its arguments are parsed."""


def _parse(argv):
    """The namespace `run.main` would have used, without running anything."""
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        captured["ns"] = real(self, args, namespace)
        raise _Stop

    with mock.patch.object(argparse.ArgumentParser, "parse_args", spy):
        try:
            run.main(argv)
        except _Stop:
            pass
    return captured["ns"]


_REQUIRED = ["--worktree", "/nonexistent/wt", "--anchor-build", "/nonexistent/build",
             "--model", "/nonexistent/model.gguf", "--store", "/nonexistent/store"]


def _referenced(code) -> set:
    """Globals and attributes a code object reaches, its nested closures included.

    `co_names` and not `co_consts`: the context key "prior_experiments" is a string
    constant and is present either way, so a check that accepted constants would pass
    against a `build_context` that had stopped calling the function entirely.
    """
    names = set(code.co_names)
    for const in code.co_consts:
        if hasattr(const, "co_names"):
            names |= _referenced(const)
    return names


class TheOptIn(unittest.TestCase):

    def test_the_cli_defaults_to_denial_four(self):
        self.assertFalse(_parse(_REQUIRED).rank_prior_experiments)

    def test_the_cli_can_turn_it_on(self):
        self.assertTrue(
            _parse([*_REQUIRED, "--rank-prior-experiments"]).rank_prior_experiments)

    def test_archive_recall_defaults_to_recency(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx.store(Path(tmp)).close()
            plain = archive.recall(Path(tmp), epoch=fx.CURRENT_EPOCH, limit=40)
            stamps = [row["recorded_at"] for row in plain]
            self.assertEqual(stamps, sorted(stamps, reverse=True))
            self.assertNotIn("magnitude_redacted", plain[0])

    def test_archive_recall_passes_the_flag_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx.store(Path(tmp)).close()
            ranked = archive.recall(Path(tmp), epoch=fx.CURRENT_EPOCH, limit=40,
                                    ranking_authorized=True)
            self.assertTrue(all(row["ranking_authorized"] for row in ranked))
            self.assertIn("magnitude_redacted", ranked[0])
            stale = [row for row in ranked if row["stale_epoch"]]
            self.assertTrue(stale, "the 40-row ranked window must reach the old epoch")
            self.assertTrue(all(row["effect_fraction"] is None for row in stale))

    def test_the_entrypoint_builds_its_history_through_that_seam(self):
        """The one call site a test cannot execute, guarded structurally.

        `build_context` is a closure inside `run.main`, and `main` verifies a real GGUF
        before it gets there, so no unit test reaches it. What a test CAN prove is that
        `main` still reaches the seam: a version that called `archive.recall` directly
        would drop `prior_experiments` from the names its code object references, and
        with it the flag.
        """
        referenced = _referenced(run.main.__code__)
        self.assertIn("prior_experiments", referenced)
        # Non-vacuity, both directions: the walk must find real references and must not
        # find things that are merely mentioned.
        self.assertIn("archive", referenced)
        self.assertNotIn("no_such_global_anywhere", referenced)

    def test_recall_always_stamps_both_provenance_markers(self):
        """What makes `render_context`'s permissive default safe on the real path.

        The pooling gate skips a row marked `stale_epoch` or explicitly marked
        non-comparable, and pools a row carrying neither. That default only ever reaches
        hand-built contexts -- provided the actual producer always stamps them, which is
        this assertion. Without it the gate would be one refactor away from fail-open on
        exactly the rows it exists to exclude.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fx.store(Path(tmp)).close()
            for authorized in (False, True):
                rows = archive.recall(Path(tmp), epoch=fx.CURRENT_EPOCH, limit=200,
                                      ranking_authorized=authorized)
                self.assertGreater(len(rows), 100)
                for row in rows:
                    self.assertIn("stale_epoch", row)
                    self.assertIn("comparable_measurement", row)
                    self.assertIs(row["comparable_measurement"],
                                  not row["stale_epoch"])

    def test_the_flag_reaches_the_store_and_not_just_the_parser(self):
        """A parsed flag that never leaves `main` authorises nothing.

        This is the seam `run.prior_experiments` exists for: a build that read the flag
        and then recalled with the authority hardcoded off passed every assertion
        written against the namespace.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fx.store(Path(tmp)).close()
            args = argparse.Namespace(store=Path(tmp), rank_prior_experiments=False)
            plain = run.prior_experiments(args, fx.CURRENT_EPOCH)
            self.assertNotIn("magnitude_redacted", plain[0])

            args.rank_prior_experiments = True
            ranked = run.prior_experiments(args, fx.CURRENT_EPOCH)
            self.assertTrue(all(row["ranking_authorized"] for row in ranked))
            self.assertNotEqual([row["attempt_id"] for row in ranked],
                                [row["attempt_id"] for row in plain])


class _StaleMagnitudeTrap(dict):
    """A cross-epoch row that raises if anyone reads its measured value."""

    def _refuse(self, key):
        if key in ex._MAGNITUDE_FIELDS:
            raise AssertionError(
                f"a CROSS-EPOCH magnitude ({key}) was read on the decision path")

    def __getitem__(self, key):
        self._refuse(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._refuse(key)
        return dict.get(self, key, default)


def _trapped_prior(limit=200):
    """Real recalled rows, with every CROSS-EPOCH magnitude wired to explode.

    Same-epoch rows are left readable on purpose. A3 permits a same-epoch magnitude to
    be read; trapping those too would make the test pass for the wrong reason, by
    proving only that nothing reads any number at all.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with fx.store(Path(tmp)) as store:
            rows = store.recall(epoch=fx.CURRENT_EPOCH, limit=limit)
    prepared = []
    for row in rows:
        if row["stale_epoch"]:
            # Put a magnitude back FIRST, so the trap is guarding a real number and not
            # an absence. Redaction is a separate defence, tested separately; here the
            # claim is that the read does not happen even when there is something to
            # read.
            row = _StaleMagnitudeTrap({**row, "effect_fraction": 0.99,
                                       "exact_attribution_effect_fraction": 0.99,
                                       "target_runtime_effect_fraction": 0.99})
        prepared.append(row)
    return prepared


class _Planner:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return loop.Hypothesis(
            mechanism_id="akm-q4k-q8-sum-sidecar", statement="s", falsifier="f",
            target_surface="ggml/src/ggml-cuda/vecdotq.cuh", target_symbol="vec_dot")

    def author(self, hypothesis, context):
        return ("ggml/src/ggml-cuda/vecdotq.cuh",)


class _Critic:
    def review_hypothesis(self, hypothesis, context):
        return loop.Review(True)

    def review_patch(self, hypothesis, paths, context):
        return loop.Review(True)


def _comparison(effect):
    return bench.Comparison(
        surface="tg128", anchor_samples=[100.0],
        candidate_samples=[100.0 * (1 + effect)], effect=effect,
        estimator="median_over_median", pairs=5, noise_floor_pct=1.0,
        residency={"invocations": 10, "resident": 10})


class TheKeepNullArithmeticCannotSeeAPriorRecord(unittest.TestCase):
    """The conformance property A3 turns on.

    Denial 4's rationale is that a reused record "would otherwise be scored against a
    floor and a threshold it was never measured under". A3 replaces the ban with a
    weight for ORDERING only; the scoring prohibition stands. So the keep/null decision
    must be provably unreachable from a prior record's number.
    """

    def _iterate(self, prior, *, effect, measure_calls=None):
        planner = _Planner()

        def measure(hypothesis, paths):
            if measure_calls is not None:
                measure_calls.append((hypothesis, paths))
            return _comparison(effect)

        return loop.iterate(
            planner=planner, critic=_Critic(),
            context={"prior_experiments": prior},
            measure=measure,
            gate=lambda h, p: (True, [gates.Verdict("compile", True)]),
            commit=lambda h, p, c: "abc1234")

    def test_a_trapped_cross_epoch_magnitude_is_never_read(self):
        prior = _trapped_prior()
        self.assertTrue([row for row in prior
                         if isinstance(row, _StaleMagnitudeTrap)],
                        "non-vacuous: the window must contain trapped stale rows")
        with self.assertRaises(AssertionError):
            prior[-1]["effect_fraction"]        # the trap is armed

        outcome = self._iterate(prior, effect=0.0001)
        self.assertEqual(outcome.status, "measured_null")
        outcome = self._iterate(prior, effect=0.05)
        self.assertEqual(outcome.status, "kept")

    def test_the_verdict_is_invariant_under_the_stale_numbers(self):
        """Same live measurement, wildly different prior magnitudes, same verdict."""
        with tempfile.TemporaryDirectory() as tmp:
            with fx.store(Path(tmp)) as store:
                rows = store.recall(epoch=fx.CURRENT_EPOCH, limit=200)
        verdicts = set()
        for poison in (None, -0.99, 0.0, 0.99, 1e9):
            prior = [dict(row, **{f: poison for f in ex._MAGNITUDE_FIELDS})
                     if row["stale_epoch"] else row for row in rows]
            outcome = self._iterate(prior, effect=0.0001)
            verdicts.add((outcome.status, outcome.comparison.effect,
                          outcome.comparison.decisive))
        self.assertEqual(len(verdicts), 1, verdicts)
        self.assertEqual(next(iter(verdicts))[0], "measured_null")

    def test_the_measurement_callable_is_handed_no_history_at_all(self):
        """The structural reason: there is no channel, not merely no traffic.

        `measure(hypothesis, paths)` takes two arguments and neither is the context.
        The comparison the keep/null test reads is therefore built from the samples the
        device produced in this run and from nothing else.
        """
        calls = []
        self._iterate(_trapped_prior(), effect=0.0001, measure_calls=calls)
        self.assertEqual(len(calls), 1)
        hypothesis, paths = calls[0]
        self.assertIsInstance(hypothesis, loop.Hypothesis)
        self.assertEqual(tuple(paths), ("ggml/src/ggml-cuda/vecdotq.cuh",))


class TheContextBundleNeverPoolsAStaleMagnitude(unittest.TestCase):
    """The defect this sweep found beyond the missing flag.

    `render_context`'s "Characterised -- do NOT re-measure these" block pooled the
    `effect_fraction` of EVERY recalled row into one median, cross-epoch rows included,
    and printed it as the reason a mechanism is finished. `stale_epoch` was already on
    those rows; the pooling loop read the number instead of the marker.

    Measured against the live store: through the first ~20 rows of epoch `6a4dccec` the
    block told the planner "`akm-q4k-q8-sum-sidecar`: measured 4x, median -8.814%" with
    all four magnitudes taken against a different anchor and build. It bit at every
    epoch transition, because a new epoch's first recall window is the old epoch's tail.
    """

    def _boundary_window(self, rows_after_transition):
        """The recall window as it stood `n` rows into the new epoch. Real rows."""
        rows = fx.rows()
        cut = [index for index, row in enumerate(rows)
               if row["epoch_sha256"] == fx.CURRENT_EPOCH][rows_after_transition - 1] + 1
        window = list(reversed(rows[:cut]))[:40]
        return [{"mechanism_id": row["mechanism_id"], "status": row["status"],
                 "statement": row["statement"], "falsifier": row["falsifier"],
                 "refusal_reason": row["refusal_reason"],
                 "effect_fraction": row["effect_fraction"],
                 "same_epoch": row["epoch_sha256"] == fx.CURRENT_EPOCH,
                 "stale_epoch": row["epoch_sha256"] != fx.CURRENT_EPOCH,
                 "comparable_measurement": row["epoch_sha256"] == fx.CURRENT_EPOCH}
                for row in window]

    def test_the_boundary_window_is_the_one_that_used_to_break_it(self):
        """NON-VACUITY. Without this the two tests below could pass on empty input."""
        window = self._boundary_window(1)
        stale = [row for row in window if row["stale_epoch"]]
        self.assertEqual(len(stale), 39)
        pooled = [row for row in stale
                  if row["mechanism_id"] == "akm-q4k-q8-sum-sidecar"
                  and row["effect_fraction"] is not None]
        self.assertGreaterEqual(len(pooled), 3,
                                "three stale samples is what used to characterise it")

    def test_a_cross_epoch_median_is_not_presented_as_characterised(self):
        text = actors.render_context({"prior_experiments": self._boundary_window(1)})
        characterised = text.split("## Characterised")[1:]
        self.assertFalse(
            any("akm-q4k-q8-sum-sidecar`: measured" in block
                for block in characterised),
            "a mechanism whose only samples are cross-epoch cannot be 'characterised'")

    def test_it_is_still_shown_as_tried_with_its_staleness_marked(self):
        """Clause 2 exactly: evidence of attempt survives, the magnitude does not."""
        # Rendered wide, because "Already tried" is itself truncated and the point here
        # is what the record SAYS, not which rows fit in a twelve-line window.
        text = actors.render_context({"prior_experiments": self._boundary_window(1)},
                                     limit=40)
        tried = text.split("## Already tried")[1]
        self.assertIn("akm-q4k-q8-sum-sidecar", tried)
        self.assertIn("STALE EPOCH", tried)

    def test_a_row_marked_not_comparable_is_not_pooled_either(self):
        """Both markers are checked, not just `stale_epoch`.

        `recall()` makes the two exact complements, so no real row can tell them apart
        and dropping this term looked free -- it survived every other assertion here.
        A producer that marks a record non-comparable for a reason other than its epoch
        would then have been silently ignored.
        """
        window = [{"mechanism_id": "akm-x", "status": "measured_null",
                   "effect_fraction": effect, "stale_epoch": False,
                   "comparable_measurement": False}
                  for effect in (-0.001, -0.002, -0.003)]
        self.assertNotIn("## Characterised",
                         actors.render_context({"prior_experiments": window}))

    def test_a_row_with_no_provenance_at_all_is_still_pooled(self):
        """The deliberate default, asserted so it cannot be tightened by accident.

        Tightening it to require a positive `comparable_measurement` switches the whole
        block off for every hand-built context -- `test_seed.py`'s five-sample run-15
        regression included. Nothing on the real path relies on this: `recall()` always
        stamps both markers, which the test above pins.
        """
        window = [{"mechanism_id": "akm-x", "status": "measured_null",
                   "effect_fraction": effect} for effect in (-0.001, -0.002, -0.003)]
        text = actors.render_context({"prior_experiments": window})
        self.assertIn("## Characterised", text)
        self.assertIn("`akm-x`: measured 3x", text)

    def test_the_characterised_block_still_fires_on_comparable_rows(self):
        """Mutation guard: deleting the feature would pass the test above."""
        window = [{"mechanism_id": "akm-x", "status": "measured_null",
                   "effect_fraction": effect, "same_epoch": True,
                   "stale_epoch": False, "comparable_measurement": True}
                  for effect in (-0.001, -0.002, -0.003)]
        text = actors.render_context({"prior_experiments": window})
        self.assertIn("## Characterised", text)
        self.assertIn("`akm-x`: measured 3x", text)

    def test_a_redacted_stale_row_cannot_be_pooled_either(self):
        """Belt and braces: rows that came through `rank()` carry no number at all."""
        with tempfile.TemporaryDirectory() as tmp:
            with fx.store(Path(tmp)) as store:
                ranked = store.recall(epoch=fx.CURRENT_EPOCH, limit=200,
                                      ranking_authorized=True)
        stale = [row for row in ranked if row["stale_epoch"]]
        self.assertGreater(len(stale), 100)
        self.assertTrue(all(row["effect_fraction"] is None for row in stale))

        text = actors.render_context({"prior_experiments": ranked}, limit=200)
        self.assertIn("STALE EPOCH", text)
        # Every mechanism the render calls characterised must have earned it from
        # comparable rows alone.
        comparable = {}
        for row in ranked:
            if row["comparable_measurement"] and row["effect_fraction"] is not None \
                    and row["mechanism_id"]:
                comparable[row["mechanism_id"]] = comparable.get(
                    row["mechanism_id"], 0) + 1
        for line in text.splitlines():
            if ": measured " in line and line.startswith("- `"):
                mechanism = line.split("`")[1]
                self.assertGreaterEqual(comparable.get(mechanism, 0), 3, line)


if __name__ == "__main__":
    unittest.main()
