"""AKX-CHK-2: a later finding supersedes an abstained/failed route without suppressing it."""

import unittest

from autokernel.loop import actors


def _row(aid, status, at, mech="m-fuse", **kw):
    return {"attempt_id": aid, "status": status, "recorded_at": at, "mechanism_id": mech,
            "same_epoch": True, "statement": f"{status} {mech}", **kw}


class AbstentionSupersession(unittest.TestCase):
    def test_derived_link_to_later_measured_row_same_mechanism(self):
        rows = [_row("a1", "abstained", "2026-10-01"), _row("k1", "keep_candidate", "2026-10-02")]
        sup = actors.abstention_supersessions(rows)
        self.assertEqual(list(sup), ["a1"])
        self.assertIs(sup["a1"], rows[1])
        self.assertEqual(len(rows), 2)  # originals untouched

    def test_no_link_for_earlier_other_mechanism_or_unmeasured_later_row(self):
        rows = [_row("a1", "abstained", "2026-10-05"),
                _row("k0", "kept", "2026-10-01"),                      # earlier
                _row("k2", "kept", "2026-10-06", mech="other"),        # other mechanism
                _row("t1", "planner_transient", "2026-10-07")]         # not a finding
        self.assertEqual(actors.abstention_supersessions(rows), {})

    def test_explicit_superseded_by_wins(self):
        rows = [_row("a1", "bench_failed", "2026-10-01", mech="", superseded_by="k9"),
                _row("k9", "measured_null", "2026-10-03", mech="x")]
        self.assertEqual(actors.abstention_supersessions(rows)["a1"]["attempt_id"], "k9")

    def test_planner_block_shows_supersession_and_frees_the_family_slot(self):
        rows = [_row("k1", "keep_candidate", "2026-10-09"),
                _row("a1", "abstained", "2026-10-01"),
                _row("a2", "abstained", "2026-10-02", mech="m-other")]
        ctx = {"prior_experiments": rows}
        text = actors.render_context(ctx, limit=1)
        self.assertIn("SUPERSEDED by k1", text)
        self.assertIn("the original record is kept", text)
        self.assertIn("abstained `m-other`", text)  # not suppressed by the superseded row

    def test_context_without_any_supersession_is_unchanged(self):
        rows = [_row("a1", "abstained", "2026-10-01"), _row("a2", "abstained", "2026-10-02", mech="z")]
        self.assertNotIn("SUPERSEDED", actors.render_context({"prior_experiments": rows}))


if __name__ == "__main__":
    unittest.main()
