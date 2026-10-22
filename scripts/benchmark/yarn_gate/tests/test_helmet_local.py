"""No-inference tests for helmet_local (metrics, json_kv generator, data readers, config record/validate)."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.yarn_gate import cells, helmet_local as hl  # noqa: E402

FAKE = lambda text: len(text) // 4  # noqa: E731
HAVE_CHECKOUT = os.path.exists(hl.HELMET_CHECKOUT + "/configs/recall.yaml")


class Metrics(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(hl.normalize_answer("The Quick, Brown (Fox)!"), "quick brown fox")

    def test_sub_em(self):
        self.assertTrue(hl.substring_exact_match_score("It was Barack Obama.", "barack obama"))
        self.assertFalse(hl.substring_exact_match_score("Obama", "Barack Obama"))
        self.assertEqual(hl.max_over_ground_truths(hl.substring_exact_match_score, "paris", ["Rome", "Paris"]), 1.0)
        self.assertTrue(hl.substring_exact_match_score("ab12-cd", "AB12CD"))  # punctuation stripped, uuid-safe

    def test_em(self):
        self.assertTrue(hl.exact_match_score("The Paris", "paris"))
        self.assertFalse(hl.exact_match_score("Paris France", "paris"))

    def test_parse_output(self):
        self.assertEqual(hl.parse_output("Answer: Foo bar\nmore"), "Foo bar")
        self.assertEqual(hl.parse_output("Foo\nbar"), "Foo")
        self.assertEqual(hl.parse_output("Corresponding value: xyz", "corresponding value:"), "xyz")

    def test_ruler_recall(self):
        self.assertEqual(hl.ruler_recall("the numbers are 123 and 456", ["123", "456", "789"]), 2 / 3)
        self.assertEqual(hl.ruler_recall("ABC", ["abc"]), 1.0)

    def test_scorers_registered_and_work(self):
        for name in ("helmet_sub_em", "helmet_kv_sub_em", "helmet_ruler_recall"):
            self.assertIn(name, cells.SCORERS)
        rag = {"scorer": "helmet_sub_em", "gold": ["Paris"], "pins": {"stop_new_line": True}, "item_id": "x"}
        self.assertEqual(cells.SCORERS["helmet_sub_em"](rag, "Answer: Paris\nRome", model="m")["score"], 1.0)
        self.assertEqual(cells.SCORERS["helmet_sub_em"](rag, "Answer: Rome\nParis", model="m")["score"], 0.0)  # stop_new_line
        kv = {"gold": ["abc-def"], "item_id": "k"}
        self.assertEqual(cells.SCORERS["helmet_kv_sub_em"](kv, " abc-def", model="m")["score"], 1.0)
        self.assertEqual(cells.SCORERS["helmet_kv_sub_em"](kv, " zzz", model="m")["score"], 0.0)


class JsonKv(unittest.TestCase):
    def test_contract_determinism_sizing(self):
        a = hl.build_cell("json_kv", 20000, 6, count=FAKE)
        b = hl.build_cell("json_kv", 20000, 6, count=FAKE)
        c = hl.build_cell("json_kv", 20000, 6, count=FAKE, seed=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0]["messages"], c[0]["messages"])
        self.assertEqual(len(a), 6)
        for i, it in enumerate(a):
            cells.validate_item(it)
            self.assertEqual(it["item_id"], f"helmet_json_kv@19k#{i:04d}")
            self.assertEqual(it["cell"], "helmet_json_kv@19k")
            self.assertEqual(it["suite"], "helmet")
            self.assertEqual(it["context_id"], it["item_id"])
            self.assertFalse(it["thinking"])
            self.assertEqual(it["max_tokens"], 100)
            self.assertEqual(it["nominal_len"], 20000)
            self.assertEqual(it["n_source"], "model_tokenizer_json")
            text = it["messages"][0]["content"]
            self.assertEqual(it["real_n_offline"], FAKE(text))
            self.assertLessEqual(it["real_n_offline"], 20000)
            self.assertGreaterEqual(it["real_n_offline"], 0.95 * 20000)
            self.assertEqual(text.count(it["gold"][0]), 1)  # gold value appears exactly once
            self.assertTrue(text.rstrip().endswith("Key: " + text.rsplit("Key: ", 1)[1]))  # question last
            self.assertIn("Extract the value corresponding", text)
            self.assertEqual(it["scorer"], "helmet_kv_sub_em")
            for k in ("helmet_repo", "helmet_sha", "config", "max_test_samples", "seed", "tokenizer_sha256", "deviations"):
                self.assertIn(k, it["pins"])
            self.assertTrue(any("LOCAL RE-GENERATION" in d for d in it["pins"]["deviations"]))
        self.assertEqual(len({it["gold"][0] for it in a}), 6)

    def test_gold_depths_cycle(self):
        items = hl.build_cell("json_kv", 16384, 6, count=FAKE)
        self.assertEqual([it["pins"]["gold_depth"] for it in items], list(hl.DEPTHS))
        # gold key is queried, and the queried key has its value paired in the JSON
        for it in items:
            text = it["messages"][0]["content"]
            key = text.rsplit("Key: ", 1)[1].strip()
            self.assertIn(f'"{key}": "{it["gold"][0]}"', text)

    def test_no_data_needed(self):
        self.assertEqual(hl.available("json_kv")[0], True)

    @unittest.skipUnless(os.path.exists(cells.DEFAULT_TOKENIZER_JSON), "tokenizer.json absent")
    def test_real_tokenizer_smoke_32k(self):
        items = hl.build_cell("json_kv", 32768, 2)
        for it in items:
            cells.validate_item(it)
            self.assertLessEqual(it["real_n_offline"], 32768)
            self.assertGreaterEqual(it["real_n_offline"], 0.95 * 32768)
            self.assertEqual(it["real_n_offline"], hl.make_counter()(it["messages"][0]["content"]))
            self.assertEqual(it["pins"]["tokenizer_sha256"], cells.DEFAULT_TOKENIZER_SHA256)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _kilt_rows(n_q: int, n_ctx: int, tag: str):
    rows = []
    for q in range(n_q):
        ctxs = [{"title": f"T{tag}{q}_{j}", "text": "lorem ipsum " * 20 + f"filler {tag}{q}_{j}", "has_answer": False}
                for j in range(n_ctx)]
        ctxs[n_ctx // 2]["text"] += f" The answer is Zorbo{q}."
        ctxs[n_ctx // 2]["has_answer"] = True
        rows.append({"id": f"{tag}{q}", "question": f"who is {tag}{q}?", "answers": [f"Zorbo{q}"], "ctxs": ctxs})
    return rows


class Readers(unittest.TestCase):
    def test_available_names_missing_paths(self):
        with tempfile.TemporaryDirectory() as d:
            ok, msg = hl.available("nq", data_dir=d)
            self.assertFalse(ok)
            self.assertIn(f"{d}/data/kilt/nq-dev-multikilt_1000_k1000_dep6.jsonl", msg)
            self.assertIn(f"{d}/data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl", msg)
            ok, msg = hl.available("hotpotqa", data_dir=d)
            self.assertIn("hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl", msg)
            ok, msg = hl.available("ruler_niah_mv", data_dir=d)
            self.assertIn("data/ruler/niah_multivalue/validation_131072.jsonl", msg)
            self.assertFalse(hl.available("nope")[0])
            with self.assertRaises(hl.HelmetDataMissing) as cm:
                hl.build_cell("nq", 131072, 2, data_dir=d, count=FAKE)
            self.assertTrue(any("k1000_dep6" in p for p in cm.exception.paths))
            with self.assertRaises(hl.HelmetDataMissing):
                hl.build_cell("ruler_niah_mk_2", 131072, 2, data_dir=d, count=FAKE)

    def test_rag_reader_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d) / "data" / "kilt"
            _write_jsonl(dd / "nq-dev-multikilt_1000_k1000_dep6.jsonl", _kilt_rows(6, 60, "d"))
            _write_jsonl(dd / "nq-train-multikilt_1000_k3_dep6.jsonl", _kilt_rows(5, 3, "t"))
            self.assertTrue(hl.available("nq", data_dir=d)[0])
            target = 3000
            items = hl.build_cell("nq", target, 4, data_dir=d, count=FAKE)
            self.assertEqual(len(items), 4)
            self.assertEqual(items, hl.build_cell("nq", target, 4, data_dir=d, count=FAKE))
            for it in items:
                cells.validate_item(it)
                text = it["messages"][0]["content"]
                self.assertLessEqual(it["real_n_offline"], target)
                self.assertGreaterEqual(it["real_n_offline"], 0.95 * target)
                self.assertTrue(text.startswith("Use the given documents"))
                self.assertIn("Document (Title: ", text)
                self.assertTrue(text.rsplit("\n\nQuestion: ", 1)[1].startswith("who is d"))  # question last
                self.assertEqual(text.count("\n\nQuestion: "), hl.SHOTS + 1)  # 2 demos + the question
                self.assertIn(it["gold"][0], text)
                self.assertEqual(it["max_tokens"], 20)
                self.assertEqual(it["scorer"], "helmet_sub_em")
                self.assertTrue(it["pins"]["stop_new_line"])

    def test_rag_240k_pads_from_k1000_file(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d) / "data" / "kilt"
            _write_jsonl(dd / "hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl", _kilt_rows(8, 20, "d"))
            _write_jsonl(dd / "hotpotqa-train-multikilt_1000_k3_dep3.jsonl", _kilt_rows(5, 3, "t"))
            # target above what one question's 20 passages hold -> padded from others
            items = hl.build_cell("hotpotqa", 8000, 2, data_dir=d, count=FAKE)
            self.assertTrue(items)
            self.assertTrue(all(it["pins"]["padded_passages"] > 0 for it in items))
            self.assertTrue(any("padded from other questions" in x for x in items[0]["pins"]["deviations"]))

    def test_rag_short_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d) / "data" / "kilt"
            _write_jsonl(dd / "nq-dev-multikilt_1000_k1000_dep6.jsonl", _kilt_rows(2, 5, "d"))
            _write_jsonl(dd / "nq-train-multikilt_1000_k3_dep6.jsonl", _kilt_rows(3, 3, "t"))
            with self.assertRaises(hl.HelmetDataShort):
                hl.build_cell("nq", 131072, 2, data_dir=d, count=FAKE)

    def test_ruler_reader_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"context": "haystack " * 100 + f"magic number for key{i} is {1000 + i}.", "query": f"key{i}",
                     "type_needle_v": "numbers", "outputs": [str(1000 + i)]} for i in range(5)]
            rows.append({"context": "x" * 400000, "query": "big", "type_needle_v": "numbers", "outputs": ["1"]})
            _write_jsonl(Path(d) / "data" / "ruler" / "niah_multikey_2" / "validation_4096.jsonl", rows)
            items = hl.build_cell("ruler_niah_mk_2", 4096, 3, data_dir=d, count=FAKE)
            self.assertEqual(len(items), 3)
            for it in items:
                cells.validate_item(it)
                text = it["messages"][0]["content"]
                self.assertTrue(text.startswith("A special magic numbers is hidden"))
                self.assertTrue(text.endswith("mentioned in the provided text?"))
                self.assertEqual(it["max_tokens"], 50)
                self.assertEqual(it["scorer"], "helmet_ruler_recall")
                self.assertEqual(it["pins"]["dropped_over_target"] <= 1, True)
            self.assertEqual(hl.build_cell("ruler_niah_mk_2", 4096, 3, data_dir=d, count=FAKE), items)
            rows2 = [dict(r, type_needle_v="numbers") for r in rows[:2]]
            _write_jsonl(Path(d) / "data" / "ruler" / "niah_multivalue" / "validation_4096.jsonl", rows2)
            mv = hl.build_cell("ruler_niah_mv", 4096, 2, data_dir=d, count=FAKE)
            self.assertTrue(mv[0]["messages"][0]["content"].startswith("Some special magic numbers are hidden"))
            self.assertEqual(hl.available("ruler_niah_mv", data_dir=d)[0], False)  # 128K file still absent


class Configs(unittest.TestCase):
    def test_parse_cell_and_k(self):
        self.assertEqual(hl.parse_cell("helmet_nq@240k#0001"), ("nq", 245760))
        self.assertEqual(hl.parse_cell("json_kv@128k"), ("json_kv", 131072))
        self.assertEqual(hl.k_for("nq", 245760), 1875)
        self.assertEqual(hl.k_for("hotpotqa", 131072), 1000)
        self.assertEqual(hl.k_for("json_kv", 245760), 3375)

    @unittest.skipUnless(HAVE_CHECKOUT, "pinned HELMET checkout absent")
    def test_record_128k_pinned(self):
        r = hl.config_record("helmet_nq@128k")
        self.assertTrue(r["path"].endswith("configs/rag.yaml"))
        self.assertEqual(r["sha256"], hl._sha256_file(r["path"]))
        self.assertEqual(r["max_test_samples"], 100)
        self.assertEqual(r["pin"], hl.HELMET_PIN)
        self.assertEqual(hl.config_record("json_kv@128k", max_test_samples=7)["max_test_samples"], 7)

    @unittest.skipUnless(HAVE_CHECKOUT, "pinned HELMET checkout absent")
    def test_local_240k_config_validates(self):
        with tempfile.TemporaryDirectory() as d:
            for task in ("nq", "json_kv"):
                cell = f"helmet_{task}@240k"
                p = hl.write_local_config(d, cell)
                cfg = json.loads(Path(p).read_text())
                group = hl.GROUP_OF[task]
                self.assertEqual(hl.validate_local_config(cfg, group), [])
                self.assertIn("_k1875_dep6", cfg["helmet_config"]["test_files"]) if group == "rag" else \
                    self.assertIn("test_k3375_dep6", cfg["helmet_config"]["test_files"])
                self.assertEqual(cfg["helmet_config"]["input_max_length"].split(",")[0], "245760")
                if group == "rag":
                    self.assertIn("_k3_dep", cfg["helmet_config"]["demo_files"])  # demo k untouched
                rec = hl.config_record(cell, out_dir=d, max_test_samples=50)
                self.assertEqual(rec["path"], p)
                self.assertEqual(rec["sha256"], hl._sha256_file(p))
                self.assertEqual(rec["kind"], "local_copy")
                self.assertEqual(hl.config_record(cell)["sha256"], rec["sha256"])  # canonical text == file
                self.assertEqual(rec["max_test_samples"], 50)

    @unittest.skipUnless(HAVE_CHECKOUT, "pinned HELMET checkout absent")
    def test_validate_rejects_tampering(self):
        base = hl.effective_config("rag", hl.LEN_240K)
        self.assertEqual(hl.validate_local_config(base, "rag"), [])
        bad = copy.deepcopy(base)
        bad["helmet_config"]["shots"] = "5"
        self.assertTrue(any("shots" in p for p in hl.validate_local_config(bad, "rag")))
        bad = copy.deepcopy(base)
        bad["helmet_config"]["test_files"] = bad["helmet_config"]["test_files"].replace("k1875", "k1000", 1)
        self.assertTrue(any("test_files" in p for p in hl.validate_local_config(bad, "rag")))
        bad = copy.deepcopy(base)
        bad["templates"]["rag_user"] += " extra"
        self.assertTrue(hl.validate_local_config(bad, "rag"))
        bad = copy.deepcopy(base)
        bad["helmet_config"]["new_key"] = "1"
        self.assertTrue(any("keys differ" in p for p in hl.validate_local_config(bad, "rag")))
        bad = copy.deepcopy(base)
        bad["helmet_config"]["input_max_length"] = "131072,131072,131072,131072"
        self.assertTrue(any("input_max_length" in p for p in hl.validate_local_config(bad, "rag")))
        bad = copy.deepcopy(base)
        bad["meta"]["pin"] = "deadbeef"
        self.assertTrue(any("pin" in p for p in hl.validate_local_config(bad, "rag")))

    @unittest.skipUnless(HAVE_CHECKOUT, "pinned HELMET checkout absent")
    def test_templates_verbatim_in_pinned_data_py(self):
        for g in ("recall", "rag"):
            self.assertEqual(hl.validate_local_config(hl.effective_config(g, hl.LEN_128K), g), [])


if __name__ == "__main__":
    unittest.main()
