import pytest

from scripts.speech.stt_bakeoff import lexicon as L

TERMS = ["EPYC", "MI210", "gfx90a", "Qwen 3 coder"]


def counts(ref, hyp, cs, terms=TERMS):
    c = L.keyword_counts(ref, hyp, terms, cs)
    return c["tp"], c["fn"], c["fp"]


# ---- 10 hand-made clips -------------------------------------------------------------
def test_clip01_exact_match():
    assert counts("we run on EPYC today", "we run on EPYC today", True) == (1, 0, 0)


def test_clip02_case_only_mismatch():
    ref, hyp = "run on MI210 today", "run on mi210 today"
    assert counts(ref, hyp, True) == (0, 1, 0)   # raw tokens differ: miss, and no hyp term span
    assert counts(ref, hyp, False) == (1, 0, 0)  # case-folded: hit


def test_clip03_substitution_is_miss():
    assert counts("run on MI210 today", "run on MI 210 today", False) == (0, 1, 0)


def test_clip04_multi_token_term_hit():
    assert counts("use Qwen 3 coder here", "use Qwen 3 coder here", True) == (1, 0, 0)


def test_clip05_multi_token_partial_is_miss():
    assert counts("use Qwen 3 coder here", "use qwen 3 code here", False) == (0, 1, 0)


def test_clip06_misaligned_correct_text_is_fp():
    # EPYC moved: min-edit aligns it as hyp-insertion + ref-deletion, so FN and FP.
    assert counts("we like EPYC chips a lot", "EPYC we like chips a lot", True) == (0, 1, 1)


def test_clip07_insertion_on_lexicon_free_text():
    rate = L.insertions_per_hour([("we ran EPYC and mi210 yesterday", 1800.0),
                                  ("nothing relevant here", 1800.0)], TERMS, False)
    assert rate == 2.0  # 2 insertions in 1 hour
    assert L.insertions_per_hour([("we ran EPYC", 1800.0)], TERMS, True) == 2.0
    assert L.insertions_per_hour([("we ran epyc", 1800.0)], TERMS, True) == 0.0


def test_clip08_non_list_wer_masks_terms():
    ref, hyp = "run EPYC fast now", "run Epyc slow now"
    assert L.non_list_wer([(ref, hyp)], TERMS, False) == pytest.approx(1 / 3)
    assert L.non_list_wer([(ref, hyp)], TERMS, True) == pytest.approx(2 / 3)


def test_clip09_repeated_terms_two_terms():
    assert counts("EPYC and MI210 and EPYC", "EPYC and MI210", True) == (2, 1, 0)


def test_clip10_duplicate_correct_hyp_is_fp():
    assert counts("gfx90a works", "gfx90a gfx90a works", True) == (1, 0, 1)


# ---- scorer extras --------------------------------------------------------------------
def test_prf_pooled_and_none_when_undefined():
    r = L.keyword_prf([("EPYC rocks", "EPYC rocks"), ("MI210 rocks", "rocks")], TERMS, True)
    assert (r["tp"], r["fn"], r["fp"]) == (1, 1, 0)
    assert r["term_precision"] == 1.0 and r["term_recall"] == 0.5
    assert r["term_f"] == pytest.approx(2 / 3)
    empty = L.keyword_prf([("hello world", "hello world")], TERMS, True)
    assert empty["term_precision"] is None and empty["term_recall"] is None and empty["term_f"] is None


def test_longest_match_first_and_non_overlapping():
    spans = L.term_spans("a Qwen 3 coder b".split(), L._term_set(["Qwen", "Qwen 3 coder"], str), str)
    assert spans == [(1, 4)]


def test_insertions_refuse_zero_duration():
    with pytest.raises(ValueError):
        L.insertions_per_hour([("EPYC", 0.0)], TERMS, True)
    with pytest.raises(ValueError):
        L.insertions_per_hour([], TERMS, True)


def test_non_list_wer_none_when_reference_all_terms():
    assert L.non_list_wer([("EPYC", "EPYC")], TERMS, True) is None


def test_score_record_fields_and_directions():
    rec = L.score_record(lexicon_sha256="a" * 64, list_scope="global", case_sensitive=False,
                         terms=TERMS, clips=[("run on MI210", "run on mi210")],
                         lexicon_free=[("plain text", 3600.0)])
    assert rec["case_policy"] == "case_folded" and rec["term_recall"] == 1.0
    assert rec["insertions_per_hour"] == 0.0 and rec["non_list_wer"] == 0.0
    assert rec["lexicon_sha256"] == "a" * 64 and rec["list_scope"] == "global"
    d = rec["metric_directions"]
    assert d["term_recall"] == d["term_precision"] == d["term_f"] == "higher_better"
    assert d["insertions_per_hour"] == d["non_list_wer"] == "lower_better"


# ---- builder --------------------------------------------------------------------------
TL = {"en": [{"term": "EPYC", "kind": "hw"}, "MI210", "gfx90a", "ab", "go",
             {"term": "Qwen 3 coder", "kind": "model", "plural": False},
             {"term": "Bach", "kind": "name"}, "EPYC"],
      "de": [{"term": "Zürich", "kind": "place"}, {"term": "Kind", "kind": "w", "plural": "Kinder"}]}


def test_builder_drops_short_and_dedups_and_reports():
    b = L.build_lexicon(TL)
    r = b["report"]
    assert r["n_terms_in"] == 10 and r["n_dropped_short"] == 2 and r["n_dropped_subword"] == 0
    assert r["n_phrases_before_expansion"] == 5 + 2  # en: EPYC MI210 gfx90a Qwen Bach (EPYC dup collapsed); de: 2
    assert r["n_phrases_after_expansion"] > r["n_phrases_before_expansion"]
    assert r["n_trie_nodes_after_expansion"] >= r["n_trie_nodes_before_expansion"]
    assert "ab" not in b["phrases"]["en"] and "go" not in b["phrases"]["en"]


def test_builder_variants_and_plural_rules():
    p = L.build_lexicon(TL)["phrases"]
    assert {"EPYC", "epyc", "Epyc", "EPYCs"} <= set(p["en"])
    assert "Baches" in p["en"] and "Bachs" not in p["en"]
    assert "Qwen 3 coder" in p["en"] and "Qwen 3 Coder" in p["en"]
    assert not any(x.endswith("coders") for x in p["en"])  # plural: false honoured
    assert "Kinder" in p["de"] and "Zürichs" not in p["de"] and "Zürichen" not in p["de"]
    assert "zürich" in p["de"]


def test_english_plural_rule():
    assert L.english_plural("box") == "boxes" and L.english_plural("Bach") == "Baches"
    assert L.english_plural("MI210") == "MI210s"


def test_builder_single_subword_allowlist():
    flag = lambda t, lang: t in {"Bach", "EPYC"}
    tl = {"en": [{"term": "Bach"}, {"term": "EPYC", "allow_single_subword": True}, "MI210"]}
    b = L.build_lexicon(tl, is_single_subword=flag)
    assert b["report"]["n_dropped_subword"] == 1
    assert "EPYC" in b["phrases"]["en"] and "Bach" not in b["phrases"]["en"]


def test_builder_deterministic_hash_and_sorted(tmp_path):
    a = L.build_lexicon(TL)
    b = L.build_lexicon({k: list(reversed(v)) for k, v in reversed(list(TL.items()))})
    assert a["lexicon_sha256"] == b["lexicon_sha256"] and len(a["lexicon_sha256"]) == 64
    assert all(v == sorted(v) for v in a["phrases"].values())
    paths = L.write_phrase_files(a, str(tmp_path))
    assert [p.rsplit("/", 1)[1] for p in paths] == ["de.txt", "en.txt"]
    assert (tmp_path / "en.txt").read_text(encoding="utf-8").splitlines() == a["phrases"]["en"]
    changed = L.build_lexicon({"en": ["EPYC"]})
    assert changed["lexicon_sha256"] != a["lexicon_sha256"]


def test_trie_nodes_custom_tokenizer():
    assert L.trie_nodes(["ab cd", "ab ef"]) == 3
    assert L.trie_nodes(["abc", "abd"], tokenize=list) == 4
