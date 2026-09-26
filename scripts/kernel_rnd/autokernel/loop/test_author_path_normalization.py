"""Author report path normalization, reply-file pickup and the lane-diff fallback on a
path mismatch.

DS41 run 10d (2026-09-25 22:39Z, thinking off): the 27B author DID edit the lane (a
79-line diff to iqk_gemm_kquants.cpp), but the iteration was lost as
`planner_transient: authoring reported ['<path>: Added an AVX-512 ...'] but the worktree
is unchanged there`, because (a) it put prose INSIDE the path string and (b) it wrote
its report to `reply.json` in the lane root (untracked), with prose on stdout.

The fixtures under `testdata/ak_run10d_author/` are the saved run-10d bytes: the lane
patch, the author's `reply.json`, its stdout and the base commit.

Pinned here:
- `<path>: <text>`, `<path> - <text>` and trailing prose resolve to `<path>` when the
  lane changed it, recorded `path_normalized: true`; never to a file it did not change;
- an untracked lane-root `reply.json` / `report.json` / `*.reply.json` is deleted before
  any gate, ignored by the lane-diff checks, and used as the report when stdout has none
  (`report_source: reply_file`);
- a report naming nothing the lane changed falls back to the lane diff (`report_source:
  lane_diff`), keeping every refusal (moved HEAD, stray untracked file, off-target diff);
- the author prompt states the stdout-only / path-only rule;
- the real run-10d replay passes `validate_candidate`.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import unittest
from unittest import mock

from autokernel.loop import actor_metrics, actors, integrity
from autokernel.loop import loop as loop_mod
from autokernel.loop.test_author_report_recovery import (TARGET, _Critic, _git,
                                                         _hypothesis, _Lane)

FIXTURES = Path(__file__).resolve().parent / "testdata" / "ak_run10d_author"
PATCH = FIXTURES / "akm-q4k-x4t-avx512.lane0.patch"
REPLY_JSON = FIXTURES / "reply.json"
STDOUT = FIXTURES / "author-stdout.txt"


def _patch_paths() -> list[str]:
    return re.findall(r"^diff --git a/(\S+) b/", PATCH.read_text(), flags=re.M)


def _preimage() -> str:
    """A synthetic pre-image of the patched file: every hunk's context and removed
    lines, in order, separated by filler so `git apply` finds each hunk by offset."""
    out: list[str] = []
    in_hunk = False
    for number, line in enumerate(PATCH.read_text().splitlines()):
        if line.startswith("@@"):
            in_hunk = True
            out.append(f"// synthetic gap {number}")
            continue
        if not in_hunk or line.startswith("\\"):
            continue
        if line.startswith((" ", "-")) or line == "":
            out.append(line[1:] if line else "")
    return "\n".join(out) + "\n"


class _RealLane(_Lane):
    """The `_Lane` harness with the target file set to the patch's pre-image."""

    def setUp(self):
        super().setUp()
        (self.ws / TARGET).write_text(_preimage())
        _git(self.ws, "add", "-A")
        _git(self.ws, "commit", "-q", "-m", "pre-image")
        self.base = _git(self.ws, "rev-parse", "HEAD")
        _git(self.ws, "checkout", "-q", "--detach", self.base)

    def apply_patch(self):
        subprocess.run(["git", "-C", str(self.ws), "apply", str(PATCH)], check=True,
                       capture_output=True)

    def author_run10d(self):
        """The run-10d author: the real edit plus the real `reply.json` at the lane root."""
        self.apply_patch()
        (self.ws / "reply.json").write_bytes(REPLY_JSON.read_bytes())


class Normalization(unittest.TestCase):

    def test_the_real_reply_entry_resolves_to_the_patched_file(self):
        [entry] = json.loads(REPLY_JSON.read_text())["paths"]
        self.assertNotEqual(entry, TARGET)
        self.assertEqual(_patch_paths(), [TARGET])
        self.assertEqual(integrity.normalize_reported_path(entry, _patch_paths()), TARGET)

    def test_the_prose_forms(self):
        changed = [TARGET, "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.h"]
        for entry in (TARGET, f"{TARGET}: text", f"{TARGET} - text", f"{TARGET} (lines 1-9)",
                      f"{TARGET}, the Q4_K body", f"  `{TARGET}`  ", f"./{TARGET}",
                      f"{TARGET}:812", f"{TARGET} — text"):
            self.assertEqual(integrity.normalize_reported_path(entry, changed), TARGET, entry)

    def test_never_a_file_the_lane_did_not_change(self):
        self.assertIsNone(integrity.normalize_reported_path(
            "ggml/src/ggml-cpu/other.cpp: text", [TARGET]))
        self.assertIsNone(integrity.normalize_reported_path(f"{TARGET}.orig", [TARGET]))
        self.assertIsNone(integrity.normalize_reported_path(f"{TARGET}: x", []))

    def test_the_longest_changed_path_wins(self):
        short, long_ = "src/a.c", "src/a.c/inner.c"
        self.assertEqual(integrity.normalize_reported_path(f"{long_}: x", [short, long_]),
                         long_)

    def test_report_artifact_names(self):
        for name in ("reply.json", "report.json", "lane0.reply.json"):
            self.assertTrue(integrity.is_report_artifact(name), name)
        for name in ("ggml/src/reply.json", "reply.json.bak", ".reply.json", "notes.txt",
                     "replies.json"):
            self.assertFalse(integrity.is_report_artifact(name), name)

    def test_the_prompt_example_carries_the_file_only(self):
        self.assertEqual(actors._surface_file(_hypothesis().target_surface), TARGET)
        self.assertEqual(actors._surface_file(TARGET), TARGET)


class Run10dReplay(_RealLane):

    def test_the_saved_patch_applies_to_the_synthetic_tree(self):
        self.apply_patch()
        self.assertEqual(integrity.dirty_paths(self.ws), (TARGET,))
        stat = _git(self.ws, "diff", "--numstat")
        self.assertTrue(stat.startswith("79\t0\t"), stat)

    def test_the_real_stdout_and_reply_file_give_the_patched_path(self):
        planner = self.planner(STDOUT.read_text(), edit=self.author_run10d, capped=False)
        got = planner.author(_hypothesis(), {})
        self.assertIsInstance(got, actors.AuthorPaths)
        self.assertEqual(tuple(got), (TARGET,))
        self.assertTrue(got.report["path_normalized"])
        self.assertEqual(got.report["report_artifacts"], ["reply.json"])
        self.assertFalse((self.ws / "reply.json").exists())        # taken out of the lane
        # ...so the real diff passes the pre-build integrity gate as declared.
        checked = integrity.validate_candidate(self.ws, got)
        self.assertEqual(checked.paths, (TARGET,))
        log = self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
        [row] = [json.loads(line) for line in log.read_text().splitlines()
                 if json.loads(line).get("schema") == actor_metrics.REPORT_SOURCE_SCHEMA]
        self.assertTrue(row["path_normalized"])
        self.assertEqual(row["paths"], [TARGET])

    def test_the_run10d_iteration_is_kept_not_lost(self):
        gate_calls = []
        outcome = self.iterate(self.planner(STDOUT.read_text(), edit=self.author_run10d,
                                            capped=False), gate_calls=gate_calls)
        self.assertEqual(outcome.status, "kept")
        self.assertEqual(gate_calls, [(TARGET,)])
        row = outcome.to_attempt()
        self.assertTrue(row["author_report_recovery"]["path_normalized"])
        self.assertIn(row["report_source"], ("reply", "reply_file"))

    def test_the_reply_file_is_the_report_when_stdout_has_none(self):
        prose = STDOUT.read_text().rsplit("\n\n", 2)[-2]           # the closing prose only
        self.assertNotIn("{", prose)
        planner = self.planner(prose, edit=self.author_run10d, capped=False)
        with mock.patch.object(actors, "_schema_repair",
                               side_effect=AssertionError("no repair turn")):
            got = planner.author(_hypothesis(), {})
        self.assertEqual(tuple(got), (TARGET,))
        self.assertEqual(got.report["report_source"], actors.REPORT_SOURCE_REPLY_FILE)
        self.assertEqual(got.report["reply_file"], "reply.json")
        self.assertTrue(got.report["path_normalized"])
        self.assertFalse((self.ws / "reply.json").exists())

    def test_an_empty_stdout_with_a_reply_file_is_not_a_transient(self):
        got = self.planner("", edit=self.author_run10d).author(_hypothesis(), {})
        self.assertEqual(got.report["report_source"], "reply_file")

    def test_the_reply_file_outcome_row(self):
        outcome = self.iterate(self.planner("", edit=self.author_run10d))
        self.assertEqual(outcome.status, "kept")
        self.assertEqual(outcome.report_source, "reply_file")

    def test_an_invalid_reply_file_is_deleted_and_never_the_report(self):
        def edit():
            self.apply_patch()
            (self.ws / "report.json").write_text('{"summary": "done"}')
            (self.ws / "lane0.reply.json").write_text("not json")
        outcome = self.iterate(self.planner("", edit=edit))
        self.assertEqual(outcome.report_source, "lane_diff")   # the fallback, not the file
        self.assertFalse((self.ws / "report.json").exists())
        self.assertFalse((self.ws / "lane0.reply.json").exists())

    def test_a_clean_reply_records_nothing_extra(self):
        def edit():
            self.apply_patch()
        got = self.planner(json.dumps({"paths": [TARGET]}), edit=edit,
                           capped=False).author(_hypothesis(), {})
        self.assertEqual(got.report["report_source"], "reply")
        self.assertFalse(got.report["path_normalized"])
        _git(self.ws, "checkout", "-q", "--", TARGET)
        outcome = self.iterate(self.planner(json.dumps({"paths": [TARGET]}), edit=edit,
                                            capped=False))
        self.assertIsNone(outcome.report_source)


class MismatchFallsBackToTheLaneDiff(_RealLane):

    def _mismatched(self, edit=None):
        return self.planner(json.dumps({"paths": ["ggml/src/ggml-cpu/elsewhere.cpp"]}),
                            edit=edit or self.apply_patch, capped=False)

    def test_the_author_raises_the_recoverable_type_on_a_mismatch(self):
        with self.assertRaises(actors.AuthorReplyMissing) as caught:
            self._mismatched().author(_hypothesis(), {})
        self.assertIn("the worktree is unchanged there", str(caught.exception))

    def test_a_mismatch_over_a_target_diff_is_recovered(self):
        gate_calls = []
        outcome = self.iterate(self._mismatched(), gate_calls=gate_calls)
        self.assertEqual(outcome.status, "kept")
        self.assertEqual(outcome.report_source, "lane_diff")
        self.assertEqual(gate_calls, [(TARGET,)])
        self.assertIn("unchanged there", outcome.author_report_recovery["reply_refusal"])

    def test_a_mismatch_without_a_reset_lane_stays_a_transient(self):
        # lane/ak-authfail-20260926: with no reset lane to prove a diff, the author's
        # misreport is an AUTHORING failure of the accepted hypothesis (kept pending).
        outcome = self.iterate(self._mismatched(), author_lane=None)
        self.assertEqual(outcome.status, loop_mod.AUTHORING_FAILED)
        self.assertIn("unchanged there", " ".join(outcome.reasons))
        self.assertEqual([c["stage"] for c in outcome.resume_checkpoints], ["author"])

    def test_a_stray_untracked_file_outside_the_allowlist_still_refuses(self):
        def edit():
            self.apply_patch()
            (self.ws / "notes.txt").write_text("scratch\n")
        outcome = self.iterate(self._mismatched(edit=edit))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertIn("untracked files outside", outcome.reasons[0])
        self.assertIn("notes.txt", outcome.reasons[0])
        self.assertIsNone(outcome.report_source)

    def test_a_stray_file_beside_a_reply_file_refuses_at_integrity(self):
        # The reply file rescues the REPORT, never the stray: the gate refuses it.
        def edit():
            self.author_run10d()
            (self.ws / "notes.txt").write_text("scratch\n")
        outcome = self.iterate(self._mismatched(edit=edit))
        self.assertEqual(outcome.status, "integrity_refused")
        self.assertEqual(outcome.integrity_screen["refusal_class"], "dirty_set_mismatch")
        self.assertIn("notes.txt", outcome.reasons[0])
        self.assertNotIn("reply.json", outcome.reasons[0])

    def test_a_moved_head_still_refuses(self):
        def edit():
            self.apply_patch()
            _git(self.ws, "commit", "-qam", "author committed")
        outcome = self.iterate(self._mismatched(edit=edit))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertIn("is not the reset base", outcome.reasons[0])

    def test_an_off_target_diff_still_refuses(self):
        outcome = self.iterate(self._mismatched(edit=lambda: self.edit(
            "x\n", path="ggml/src/ggml-cpu/other.cpp")))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertIn("is the hypothesis's target surface", outcome.reasons[0])

    def test_lane_diff_report_ignores_a_reply_file(self):
        self.author_run10d()
        self.assertEqual(integrity.lane_diff_report(self.ws, self.base,
                                                    target_surface=_hypothesis().target_surface),
                         (TARGET,))


class AuthorPrompt(_Lane):

    def test_the_prompt_states_the_stdout_and_path_only_rule(self):
        self.planner(json.dumps({"paths": [TARGET]}), edit=self.edit,
                     capped=False).author(_hypothesis(), {"backend": "cpu"})
        prompt = actors._run_agent.call_args.args[0]
        self.assertIn(actors.AUTHOR_REPLY_RULE, prompt)
        self.assertIn("do not write it", actors.AUTHOR_REPLY_RULE)
        self.assertIn("no description", actors.AUTHOR_REPLY_RULE)


if __name__ == "__main__":
    unittest.main()
