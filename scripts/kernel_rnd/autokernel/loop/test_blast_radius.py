"""AKX-P1e: the production classifier assembles a screen the real writer accepts."""
import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from autokernel.evaluator import blast_radius_controls as C
from autokernel.loop import blast_radius as BR, keep_screen as KS


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


class ScreenKeepTests(unittest.TestCase):
    def setUp(self):
        self.fp = C.produce(C.Q8_0_DISPATCH)
        self.dso = {"libggml-cpu.so": sha("dso")}
        self.workloads = [
            C.workload("W3", is_author_target=True),
            C.workload("W4", t0_dso_anchor=self.dso, t0_dso_candidate=dict(self.dso)),
            C.workload("W6", census_stale=True)]
        self.keys = {w.workload_id: sha(w.workload_id) for w in self.workloads}

    def screen(self):
        return BR.screen_keep(self.fp, self.workloads, keep_id="keep-1", census_freshness_keys=self.keys,
                              created_at=datetime.now(timezone.utc).isoformat())

    def test_rows_pass_the_production_contract_and_unknown_decides_touched(self):
        screen, rows = self.screen()
        self.assertEqual(screen["touched_ids"], ["W3"])
        self.assertEqual(screen["inert_ids"], ["W4"])
        self.assertEqual(screen["unknown_ids"], ["W6"])
        for row in rows:
            self.assertTrue(KS._valid_row(row, keep_id="keep-1", workload_id=row["workload_id"]))
        self.assertEqual({r["workload_id"]: r["decision_class"] for r in rows},
                         {"W3": "TOUCHED", "W4": "INERT", "W6": "TOUCHED"})

    def test_the_real_writer_publishes_the_screen(self):
        screen, rows = self.screen()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "keep.diff"
            source.write_text("x\n", encoding="utf-8")
            target = KS.write_record(tmp, keep_id="keep-1", campaign_id="c", campaign_manifest_sha256=sha("m"),
                                     source_path=str(source), source_sha256=hashlib.sha256(b"x\n").hexdigest(),
                                     workload_signatures={k: sha("sig" + k) for k in self.keys},
                                     observed_at=datetime.now(timezone.utc).isoformat(), screen=screen, rows=rows)
            self.assertTrue(target.is_file())

    def test_a_workload_without_a_census_key_is_refused(self):
        del self.keys["W6"]
        with self.assertRaisesRegex(ValueError, "W6"):
            self.screen()

    def test_an_empty_workload_set_is_refused(self):
        with self.assertRaises(ValueError):
            BR.screen_keep(self.fp, [], keep_id="keep-1", census_freshness_keys={}, created_at="t")


if __name__ == "__main__":
    unittest.main()
