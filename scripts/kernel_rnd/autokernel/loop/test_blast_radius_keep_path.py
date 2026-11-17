"""AKX-P1e: the classifier through the real keep path (record -> write_record -> project).

LIVE_FOLLOWUP: censuses and T0/T1 witnesses here are recorded fixtures (P1d); real ones
come from AKX-P0d censuses and live builds/forwards.
"""
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from ..evaluator import blast_radius_controls as C
from ..evaluator import footprint as FP
from . import blast_radius as BR
from . import keep_screen as KS

ALL = ("W1", "W2", "W3", "W4", "W5", "W6")
DSO = {"libggml-base.so": C.sha("base"), "libggml-cpu.so": C.sha("cpu")}
T1 = (C.sha("cov-anchor"), C.sha("cov-cand"), C.sha("envelope"))


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _Store(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manifest = self.root / "campaign.json"
        self.manifest.write_text('{"campaign": "fixture"}', encoding="utf-8")
        self.digest = C.sha("manifest")
        self.targets = [{"target_ids": [w], "workload_signature": C.sha("sig-" + w),
                         "status": "ready"} for w in ALL]

    def tearDown(self):
        self._tmp.cleanup()

    def record(self, keep, fp, censuses=None):
        path = BR.record_keep_screen(
            self.root, keep_id=keep, resolved_targets=self.targets, campaign_id="c",
            manifest_digest=self.digest, manifest_path=str(self.manifest),
            footprint=fp, censuses=censuses, observed_at=_now())
        self.assertEqual(path, KS.record_path(self.root, keep))
        return self.project(keep)

    def project(self, keep):
        env = KS.project(self.root, keeps=[keep], resolved_targets=self.targets,
                         campaign_id="c", manifest_digest=self.digest,
                         manifest_path=str(self.manifest))
        (entry,) = env["keeps"]
        self.assertEqual(entry["freshness"]["state"], "fresh", entry["freshness"])
        return {r["workload_id"]: r for r in entry["rows"]}

    def assert_valid_rows(self, keep):
        body = KS._read_record(KS.record_path(self.root, keep))
        for row in body["rows"]:
            self.assertTrue(KS._valid_row(row, keep_id=keep, workload_id=row["workload_id"]))


def census(wid, **kw):
    types = C.W5_WITHOUT_Q8 if wid == "W5" else None
    return (C.workload(wid, types=types, **kw), C.sha("census-" + wid))


class NoCensusUnchanged(_Store):
    """(a) no census -> every workload UNKNOWN (TOUCHED): the required set is unchanged."""

    def _assert_all_required(self, rows, keep):
        self.assertEqual(set(rows), set(ALL))
        self.assertTrue(all(r["classification"] == "UNKNOWN" for r in rows.values()))
        body = KS._read_record(KS.record_path(self.root, keep))
        self.assertEqual(body["screen"]["inert_ids"], [])
        self.assertTrue(all(r["decision_class"] == "TOUCHED" for r in body["rows"]))
        self.assert_valid_rows(keep)
        # Same classification set as a keep that has no record at all (pre-P1e).
        absent = KS.project(self.root, keeps=["keep-none"], resolved_targets=self.targets,
                            campaign_id="c", manifest_digest=self.digest,
                            manifest_path=str(self.manifest))["keeps"][0]["rows"]
        self.assertEqual({r["workload_id"]: r["classification"] for r in absent},
                         {w: r["classification"] for w, r in rows.items()})

    def test_every_patch_kind_without_census(self):
        for i, patch in enumerate((C.COMMENT_ONLY, C.Q8_0_DISPATCH,
                                   C.KNOB_READ_NO_DEFAULT_CHANGE, C.KNOB_DEFAULT_FLIP_ON)):
            keep = f"keep-a{i}"
            self._assert_all_required(self.record(keep, C.produce(patch)), keep)

    def test_no_footprint(self):
        self._assert_all_required(self.record("keep-nofp", None), "keep-nofp")

    def test_loop_hook_on_a_real_commit(self):
        repo = self.root / "repo"
        repo.mkdir()
        git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b",
                                         "-c", "user.name=a", *a], check=True,
                                        capture_output=True, text=True).stdout.strip()
        git("init", "-q")
        (repo / "k.c").write_text("int f(void) { return 0; }\n")
        git("add", "k.c")
        git("commit", "-qm", "base")
        (repo / "k.c").write_text("// note\nint f(void) { return 0; }\n")
        git("commit", "-qam", "keep")
        campaign = SimpleNamespace(
            targets=[SimpleNamespace(to_dict=lambda t=t: t) for t in self.targets],
            campaign_id="c", manifest_digest=self.digest)
        path = BR.record_keep_best_effort(self.root, repo, keep_id="keep-hook",
                                          child=git("rev-parse", "HEAD"),
                                          resolved_campaign=campaign,
                                          manifest_path=str(self.manifest))
        self.assertIsNotNone(path)
        self._assert_all_required(self.project("keep-hook"), "keep-hook")

    def test_loop_hook_without_campaign_is_a_no_op(self):
        self.assertIsNone(BR.record_keep_best_effort(
            self.root, self.root, keep_id="k", child="HEAD", resolved_campaign=None,
            manifest_path=None))


class P1dControlsThroughKeepPath(_Store):
    """(b) the P1d controls, published and re-read through the real record/project path."""

    def test_comment_only_inert_by_t0_everywhere(self):
        cs = {w: census(w, t0_dso_anchor=DSO, t0_dso_candidate=dict(DSO)) for w in ALL}
        rows = self.record("keep-comment", C.produce(C.COMMENT_ONLY), cs)
        for w in ALL:
            self.assertEqual((rows[w]["classification"], rows[w]["tier"]), ("INERT", "T0"), w)
        self.assert_valid_rows("keep-comment")

    def test_q8_dispatch_touched_w1_w3_inert_w2_w4_w6(self):
        cs = {w: census(w, t1_profiles=T1) for w in ALL}
        rows = self.record("keep-q8", C.produce(C.Q8_0_DISPATCH), cs)
        for w in ("W1", "W3"):
            self.assertEqual(rows[w]["classification"], "TOUCHED", w)
        for w in ("W2", "W4", "W6", "W5"):  # W5 census here lacks Q8_0
            self.assertEqual((rows[w]["classification"], rows[w]["tier"]), ("INERT", "T1"), w)
        self.assert_valid_rows("keep-q8")

    def test_knob_read_no_default_change_inert_where_unset(self):
        fp = C.produce(C.KNOB_READ_NO_DEFAULT_CHANGE)
        knob = fp.env_knobs[0].name
        cs = {w: census(w, t1_profiles=T1) for w in ALL}
        cs["W3"] = census("W3", t1_profiles=T1, effective_env={knob: "1"})
        rows = self.record("keep-knob", fp, cs)
        self.assertEqual(rows["W3"]["classification"], "TOUCHED")
        for w in ("W1", "W2", "W4", "W5", "W6"):
            self.assertEqual(rows[w]["classification"], "INERT", w)
        self.assert_valid_rows("keep-knob")

    def test_inert_needs_census_proof(self):
        # Mixed: only W2 has a census; everyone else stays UNKNOWN (required).
        rows = self.record("keep-mixed", C.produce(C.Q8_0_DISPATCH),
                           {"W2": census("W2", t1_profiles=T1)})
        self.assertEqual(rows["W2"]["classification"], "INERT")
        for w in ("W1", "W3", "W4", "W5", "W6"):
            self.assertEqual(rows[w]["classification"], "UNKNOWN", w)

    def test_stale_census_unknown(self):
        cs = {w: census(w, census_stale=True, t0_dso_anchor=DSO, t0_dso_candidate=dict(DSO))
              for w in ALL}
        rows = self.record("keep-stale", C.produce(C.COMMENT_ONLY), cs)
        self.assertTrue(all(r["classification"] == "UNKNOWN" for r in rows.values()))

    def test_empty_footprint_refuses_and_writes_nothing(self):
        fp = C.produce(C.COMMENT_ONLY)
        empty = FP.PatchFootprint(**{**fp.__dict__, "hunks": (), "dispatch_sites": ()})
        cs = {w: census(w) for w in ALL}
        with self.assertRaises(FP.FootprintError):
            BR.record_keep_screen(self.root, keep_id="keep-empty",
                                  resolved_targets=self.targets, campaign_id="c",
                                  manifest_digest=self.digest,
                                  manifest_path=str(self.manifest), footprint=empty,
                                  censuses=cs)
        self.assertFalse(KS.record_path(self.root, "keep-empty").exists())


if __name__ == "__main__":
    unittest.main()
