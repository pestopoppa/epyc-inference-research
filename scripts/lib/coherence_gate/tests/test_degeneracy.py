"""degeneracy.v2 (and frozen inf70-degeneracy.v1). Ported from gpu-block-27b-20261003/test_degeneracy.py
(18 tests, 2026-10-04) onto vendored fixtures. No inference.

    python3 -m pytest -q scripts/lib/coherence_gate/tests
"""
from __future__ import annotations

import random

import pytest
from conftest import PROSE, STORED, TRACES, V1_GOLDEN, cut, enc, needs_tokenizer, TOK

from coherence_gate import degeneracy as D


# ------------------------------------------------------------------ coherent long text passes v2
@needs_tokenizer
@pytest.mark.parametrize("n", [1000, 1500])
def test_long_model_reasoning_is_ok_in_v2_and_salad_in_v1(n):
    assert len(TRACES) >= 5
    for r in TRACES:
        text, ids = cut(r, n)
        v2 = D.classify(text, ids, "length")
        assert v2["cls"] == "OK", v2
        assert v2["semantic"] == "unchecked" and v2["classifier"] == D.V2_ID
        # the defect being fixed: v1 calls every one of these SALAD on uniq alone
        v1 = D.classify_v1(text, ids, "length")
        assert v1["cls"] == "SALAD" and v1["salad_reasons"] == ["uniq"], v1


@needs_tokenizer
def test_long_prose_is_ok():
    text, ids = cut(PROSE, 1500)
    assert D.classify(text, ids, "length")["cls"] == "OK"


# ------------------------------------------------------------------ genuine degeneracy still fails
def _assert_degenerate(text, ids, want):
    c = D.classify(text, ids, "length")
    assert c["cls"] == "DEGENERATE" and D.fails(c), c
    assert want & set(c["reasons"]), c
    return c


@needs_tokenizer
def test_stuck_token_run():
    text, ids = cut(PROSE[:2000] + " the" * 40, 600)
    _assert_degenerate(text, ids, {"run", "loop"})


@needs_tokenizer
def test_short_phrase_loop():
    text, ids = cut("I need to check this again. " * 200, 1500)
    _assert_degenerate(text, ids, {"loop", "top"})


@needs_tokenizer
def test_long_period_paragraph_loop_with_low_top_and_run():
    para = " ".join(PROSE.split()[200:310]) + "\n\n"   # ~150-token paragraph, repeated
    text, ids = cut(para * 12, 1500)
    c = _assert_degenerate(text, ids, {"loop"})
    assert c["top"] < D.TOP_MAX and c["run"] < D.RUN_MAX     # top/run alone would miss it
    assert c["loop"]["period"] > 100


@needs_tokenizer
def test_late_onset_loop_after_coherent_reasoning():
    head = TOK.decode(enc(TRACES[0])[:700])
    text, ids = cut(head + " Wait, let me re-check the timezone conversion once more." * 60, 1500)
    c = _assert_degenerate(text, ids, {"loop"})
    assert c["loop"]["at_token"] >= 650


@needs_tokenizer
def test_incrementing_counter_loop():
    body = "".join(f"Step {k}: add one to the counter and check the result again.\n" for k in range(1, 200))
    text, ids = cut(body, 1500)
    _assert_degenerate(text, ids, {"loop"})


@needs_tokenizer
def test_small_vocabulary_word_soup():
    rng = random.Random(7)
    words = ["model", "kernel", "the", "decode", "value", "token"]
    text, ids = cut(" ".join(rng.choice(words) for _ in range(1500)), 1500)
    c = _assert_degenerate(text, ids, {"uniq+stuck", "top", "run"})
    assert "uniq_low" in c["soft"]


@needs_tokenizer
def test_random_token_garbage():
    rng = random.Random(11)
    ids = [rng.randrange(0, 150000) for _ in range(800)]
    text = TOK.decode(ids)
    ids = enc(text)
    c = D.classify(text, ids, "length")
    # never a silent OK: DEGENERATE on ascii; an ascii-ONLY trigger is `review` (eyeball required)
    assert c["cls"] == "DEGENERATE" and "ascii" in c["reasons"], c
    assert D.fails(c) or c["review"]


@needs_tokenizer
def test_shuffled_words_are_out_of_scope():
    """Large-vocabulary word soup has natural token statistics: no token-stream detector can call it.
    v2 says OK with semantic 'unchecked' - this pins the documented limit, it is not a pass of meaning."""
    w = PROSE.split()[:1200]
    random.Random(3).shuffle(w)
    text, ids = cut(" ".join(w), 1200)
    c = D.classify(text, ids, "length")
    assert c["cls"] == "OK" and c["semantic"] == "unchecked"


# ------------------------------------------------------------------ uniq is corroborating only
def test_uniq_low_alone_never_fires():
    c = D.classify_stats({"n": 1500, "uniq": 0.05, "top": 0.05, "run": 2, "words": 700, "ascii_ok": 0.97})
    assert c["cls"] == "OK" and "uniq_low" in c["soft"]
    c = D.classify_stats({"n": 1500, "uniq": 0.05, "top": 0.12, "run": 2, "words": 700, "ascii_ok": 0.97})
    assert c["cls"] == "DEGENERATE" and c["reasons"] == ["uniq+stuck"]


def test_uniq_floor_is_length_aware():
    assert D.uniq_floor(100) == D.uniq_floor(200) == pytest.approx(0.30)
    assert D.uniq_floor(800) == pytest.approx(0.15)
    assert D.uniq_floor(1500) < 0.12 < D.uniq_floor(1000) < 0.14


# ------------------------------------------------------------------ short outputs
@needs_tokenizer
def test_correct_one_token_answer_is_short_not_failure():
    c = D.classify("E", enc("E"), "stop")
    assert c["cls"] == "SHORT" and c["eos_short"] and not D.fails(c)
    assert c["v1_cls"] == "EARLY-EOS"


def test_empty_eos_and_http_error_fail():
    assert D.fails(D.classify("", [], "stop"))
    assert D.fails(D.classify("", [], None))
    assert D.fails(D.classify("x", [1], "stop", http_ok=False))


@needs_tokenizer
def test_ascii_only_is_review_not_fail():
    code = "def f(x):\n    return {k: v for k, v in x.items() if k != '_'}\n"
    code = code + "".join(f"_tbl[{i}] = <{i}> => {{{i}}};\n" for i in range(40))
    c = D.classify(code, enc(code), "stop")
    if c["reasons"] == ["ascii"]:
        assert c["review"] and not D.fails(c)


# ------------------------------------------------------------------ v1 stays frozen
@needs_tokenizer
def test_v1_identical_to_lib_gpublock_golden():
    """v1 vs the lib_gpublock.classify outputs captured by fixtures/make_fixtures.py."""
    samples = {"prose1500": cut(PROSE, 1500), "trace1000": cut(TRACES[0], 1000), "ok30": cut("ok " * 30, 30),
               "E": ("E", enc("E"))}
    for name, (text, ids) in samples.items():
        for fin in ("stop", "length"):
            b = D.classify_v1(text, ids, fin)
            b.pop("classifier")
            assert b == V1_GOLDEN[name][fin], (name, fin)


# ------------------------------------------------------------------ the Q38-T7 rows
def test_q38t7_stored_rows_rescore():
    gen = [r for r in STORED if r["phase"] in ("B", "C")]
    assert len(gen) == 12 and sum(r["coherence"]["cls"] == "SALAD" for r in gen) == 11
    for r in gen:
        assert D.classify_stats(r["coherence"])["cls"] == "OK"
    gsm = [r for r in STORED if r["phase"] == "A" and r["id"].startswith("gsm8k_00")]
    assert gsm
    for r in gsm:
        assert D.classify_stats(r["coherence"])["cls"] == "OK"
