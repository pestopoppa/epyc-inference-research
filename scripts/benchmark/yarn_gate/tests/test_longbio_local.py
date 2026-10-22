"""Offline tests for longbio_local (fake whitespace counter; one real-tokenizer smoke, no server)."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.yarn_gate import cells, longbio_local as lb  # noqa: E402

OK, WHY = lb.available(check_tokenizer=False)
fake = lambda text: len(text.split())  # noqa: E731
T = 3000


@unittest.skipUnless(OK, WHY)
class LongBioTests(unittest.TestCase):
    def test_determinism_and_contract(self):
        a = lb.build_cell("standard", T, 3, count=fake)
        b = lb.build_cell("standard", T, 3, count=fake)
        self.assertEqual(a, b)
        c = lb.build_cell("standard", T, 3, count=fake, seed=1)
        self.assertNotEqual(a[0]["messages"], c[0]["messages"])
        self.assertEqual(len({i["item_id"] for i in a}), 3)
        for it in a:
            cells.validate_item(it)
            self.assertEqual(it["item_id"], f"longbio_standard@2k#{a.index(it):04d}")
            self.assertEqual(it["cell"], "longbio_standard@2k")
            self.assertEqual((it["suite"], it["task"], it["thinking"]), ("longbio", "standard", False))
            self.assertEqual(it["messages"][-1]["role"], "user")
            self.assertEqual(it["pins"]["longbio_sha"], lb.REPO_SHA)

    def test_target_gold_question_and_sizing_all_variants(self):
        for v in lb.VARIANTS:
            for it in lb.build_cell(v, T, 2, count=fake):
                dp, user = it["datapoint"], it["messages"][-1]["content"]
                ctx, _, q = user.partition("\n\nQuestion: ") if dp["retrievals"] == 1 else \
                    user.partition("\n\nThe Questions are as follows:\n")
                self.assertTrue(user.startswith("Context: "))
                self.assertEqual(len(it["gold"]), dp["retrievals"])
                self.assertTrue(user.rstrip().endswith(q.strip()) or "Answer each question" in user)
                for name, gold, question in zip(dp["names"], it["gold"], dp["questions"]):
                    self.assertIn(question, user)
                    self.assertIn(name, question)
                    # the person's bio states the gold value (bio is "Below is the bio of NAME. ... ")
                    bio_start = ctx.index(f"of {name}.")
                    self.assertIn(gold, ctx[bio_start:bio_start + 1500].split("Below is")[0] + " ")
                self.assertEqual(ctx.count(f"Below is the bio of {dp['names'][0]}.") +
                                 ctx.count(f"Below is the self-introduction of {dp['names'][0]}."), 1)
                n = fake(user)
                self.assertEqual(n, it["real_n_offline"])
                self.assertLessEqual(n, T)
                self.assertGreaterEqual(n, 0.95 * T, (v, n))
                self.assertEqual(it["n_source"], "injected_counter")

    def test_paraphrase_differs_from_standard(self):
        s = lb.build_cell("standard", T, 1, count=fake)[0]["messages"][1]["content"]
        p = lb.build_cell("paraphrase", T, 1, count=fake)[0]["messages"][1]["content"]
        self.assertIn("Below is the bio of", p)
        self.assertNotEqual(s, p)

    def test_scorer(self):
        one = lb.build_cell("standard", T, 1, count=fake)[0]
        g = one["gold"][0]
        sc = lambda it, o: cells.score(it, o, model="m")["score"]  # noqa: E731
        self.assertEqual(sc(one, g), 1.0)
        self.assertEqual(sc(one, f"  {g.upper()}.  "), 1.0)
        self.assertEqual(sc(one, "zzz nothing"), 0.0)
        m = lb.build_cell("multi_standard_5", T, 1, count=fake)[0]
        gold = m["gold"]
        self.assertEqual(len(gold), 5)
        self.assertEqual(sc(m, "\n".join(f"{i + 1}. {x}" for i, x in enumerate(gold))), 1.0)
        part = "\n".join(gold[:3] + ["unknownzzz", "unknownyyy"])
        self.assertAlmostEqual(sc(m, part), 3 / 5 if len(set(gold)) == 5 else sc(m, part))
        self.assertEqual(sc(m, "nothing"), 0.0)
        # order matters (repo: sequential search)
        if len(set(gold)) == 5:
            self.assertLess(sc(m, "\n".join(reversed(gold))), 1.0)

    def test_available_names_missing_path(self):
        old = lb.REPO_DIR
        try:
            lb.REPO_DIR = "/nonexistent/longbio/clone"
            ok, why = lb.available(check_tokenizer=False)
            self.assertFalse(ok)
            self.assertIn("/nonexistent/longbio/clone", why)
        finally:
            lb.REPO_DIR = old
        ok, why = lb.available("/nonexistent/tok.json")
        self.assertFalse(ok)
        self.assertIn("/nonexistent/tok.json", why)

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            lb.build_cell("nope", T, 1, count=fake)


@unittest.skipUnless(lb.available()[0], lb.available()[1])
class RealTokenizerSmoke(unittest.TestCase):
    def test_32k_real_tokens(self):
        t0 = time.time()
        it = lb.build_cell("standard", 32768, 1)[0]
        dt = time.time() - t0
        print(f"\n[longbio smoke] 32768 build {dt:.1f}s real_n={it['real_n_offline']}")
        self.assertGreaterEqual(it["real_n_offline"], 0.95 * 32768)
        self.assertLessEqual(it["real_n_offline"], 32768)
        self.assertEqual(it["n_source"], "model_tokenizer_json")
        self.assertEqual(it["pins"]["tokenizer_sha256"], cells.DEFAULT_TOKENIZER_SHA256)


if __name__ == "__main__":
    unittest.main()
