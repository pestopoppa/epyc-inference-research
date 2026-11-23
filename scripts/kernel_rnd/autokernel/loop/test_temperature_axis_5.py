"""AK-TEMP-AXIS-5: the DICE replay as an offline temperature re-scoring instrument."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from autokernel.loop import sampling_axis as S, spec_temperature_gate as G, temperature_rescore as R

CAPTURE_DIR = Path("/mnt/raid0/llm/tmp/ak-dice-capture-a3b-mtp-cpu-20261123-r2/out")
DICE_DIR = R.DICE_REPLAY.parent
LOGITS = [(10, 5.0), (11, 4.0), (12, 3.0), (13, -9.0)]


def synthetic(tmp: Path) -> tuple[Path, list[dict]]:
    """One 2-draft round: chain tokens 10 then 11; target logits as LOGITS at every row."""
    rec = {"k": "round", "v": "dice.v1", "task": 0, "round": 1, "T": 0.5, "top_k": 40, "top_p": 0.95,
           "min_p": 0.05, "D": 2, "chain": [10, 11],
           "mtp_top": [[[10, 5.0, 0.7], [11, 4.0, 0.3]], [[11, 5.0, 0.8], [10, 3.0, 0.2]]],
           "rows": [[[list(x) for x in LOGITS], [[10, 1.0]]] for _ in range(3)]}
    blob = (json.dumps(rec) + "\n").encode()
    path = tmp / "cap.jsonl"
    path.write_bytes(blob)
    unit = {"unit_id": "u0", "prompt_idx": 0, "T": 0.5, "off0": 0, "off1": len(blob),
            "check": {"top_k": 40, "top_p": 0.95, "min_p": 0.05}}
    return path, [unit]


class SamplerModel(unittest.TestCase):
    kw = dict(top_k=40, top_p=0.95, min_p=0.05)

    def test_zero_is_a_point_mass_on_the_argmax(self):
        self.assertEqual(R.sampler_probs(LOGITS, 0, **self.kw), {10: 1.0})

    def test_candidate_set_is_independent_of_temperature_and_sums_to_one(self):
        sets = set()
        for t in (0.1, 0.2, 0.3, 1.0):
            p = R.sampler_probs(LOGITS, t, **self.kw)
            self.assertAlmostEqual(sum(p.values()), 1.0)
            sets.add(frozenset(p))
        self.assertEqual(len(sets), 1)
        self.assertNotIn(13, next(iter(sets)))  # min_p / top_p cut the -9 logit

    def test_lower_temperature_concentrates_mass(self):
        hot, cold = (R.sampler_probs(LOGITS, t, **self.kw) for t in (1.0, 0.2))
        self.assertGreater(cold[10], hot[10])
        self.assertAlmostEqual(hot[10] / hot[11], math.exp(1.0), places=9)

    def test_top_k_and_min_p_filter(self):
        self.assertEqual(set(R.sampler_probs(LOGITS, 1.0, top_k=1, top_p=1.0, min_p=0.0)), {10})
        self.assertEqual(set(R.sampler_probs(LOGITS, 1.0, top_k=40, top_p=1.0, min_p=0.3)), {10, 11})


class Synthetic(unittest.TestCase):
    def test_rescore_reproduces_the_hand_computed_expectations(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, units = synthetic(Path(tmp))
            dice, _ = R.load_dice()
            out = R.rescore(cap, units, (0.0, 1.0), ws=(1, 2), dice=dice)
        by = {(r["T"], r["W"]): r for r in out["rows"]}
        # T = 0: argmax is token 10 at every row, chain is (10, 11): one accepted draft, then a reject.
        self.assertAlmostEqual(by[(0.0, 1)]["V0"], 2.0)
        self.assertAlmostEqual(by[(0.0, 2)]["V0"], 2.0)
        p = R.sampler_probs(LOGITS, 1.0, top_k=40, top_p=0.95, min_p=0.05)
        self.assertAlmostEqual(by[(1.0, 1)]["V0"], 1 + p[10])
        self.assertAlmostEqual(by[(1.0, 2)]["V0"], 1 + p[10] + p[10] * p[11])
        self.assertTrue(all(r["V2"] >= r["V0"] - 1e-12 for r in out["rows"]))

    def test_output_carries_projection_arms_in_the_sampling_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, units = synthetic(Path(tmp))
            out = R.rescore(cap, units, R.DEFAULT_TEMPS, ws=(1,), dice=R.load_dice()[0])
        arms = S.arms(out["sampling"])
        self.assertEqual([a["temperature"] for a in arms], [0.0, 0.2, 1.0])
        self.assertTrue(all(a["kind"] == "projection" for a in arms))
        # a projection never satisfies the served-temperature gate
        d = G.decide(out["sampling"], [], (0.1, 1.0))
        self.assertFalse(d["ok"])
        self.assertIn("unknown", d["reason"])

    def test_source_temperature_is_exact_else_nearest_lower_on_ties(self):
        self.assertEqual(R.source_temperature(0.3, [0.1, 0.3, 1.0]), 0.3)
        self.assertEqual(R.source_temperature(0.2, [0.1, 0.3, 1.0]), 0.1)
        self.assertEqual(R.source_temperature(0.0, [0.1, 0.3, 1.0]), 0.1)

    def test_cli_writes_the_record_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, units = synthetic(Path(tmp))
            idx = Path(tmp, "units.jsonl")
            idx.write_text("\n".join(json.dumps(u) for u in units), encoding="utf-8")
            out = Path(tmp, "out.json")
            self.assertEqual(R.main(["--capture", str(cap), "--units", str(idx), "--temps", "0", "1",
                                     "--out", str(out)]), 0)
            self.assertEqual(json.loads(out.read_text())["schema"], R.SCHEMA)


@unittest.skipUnless((CAPTURE_DIR / "capture" / "cap-L01.jsonl").exists(),
                     "DICE A3B MTP capture not on this host")
class AgainstTheCommittedReplay(unittest.TestCase):
    """The instrument must reproduce research 03c100462's committed RESULT.json."""
    cap = CAPTURE_DIR / "capture" / "cap-L01.jsonl"

    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((DICE_DIR / "RESULT.json").read_text(encoding="utf-8"))
        cls.units = R.load_units(CAPTURE_DIR / "results.jsonl")

    def test_the_replay_script_is_the_one_the_result_was_made_with(self):
        _, sha = R.load_dice()
        self.assertEqual(sha, self.result["digests"]["script_dice_replay.py"])

    def test_the_sampler_model_is_the_served_one(self):
        out = R.rescore(self.cap, self.units, (0.1, 0.3, 1.0), prompts=[32], ws=(1,),
                        dice=R.load_dice()[0])
        self.assertLess(out["sampler_check_max_deviation_at_capture_T"], 5e-4)

    def test_holdout_V0_and_V2_match_the_committed_table_at_the_capture_temperatures(self):
        out = R.rescore(self.cap, self.units, (0.1, 0.3, 1.0), prompts=R.HOLDOUT_PROMPTS)
        mine = {(r["T"], r["W"]): r for r in out["rows"]}
        for row in self.result["table_holdout_prompts_32_63"]:
            got = mine[(row["T"], row["W"])]
            self.assertEqual(got["rounds"], row["rounds_holdout"])
            self.assertAlmostEqual(got["V0"], row["V0"], places=9)
            self.assertAlmostEqual(got["V2"], row["V2"], places=9)

    def test_rescoring_at_new_temperatures_is_defined_and_in_range(self):
        out = R.rescore(self.cap, self.units, (0.0, 0.2, 1.0), prompts=R.HOLDOUT_PROMPTS, ws=(4,))
        self.assertEqual(sorted({r["T"] for r in out["rows"]}), [0.0, 0.2, 1.0])
        for r in out["rows"]:
            self.assertTrue(1.0 <= r["V0"] <= 5.0 and r["V0"] <= r["V2"] + 1e-9, r)


if __name__ == "__main__":
    unittest.main()
