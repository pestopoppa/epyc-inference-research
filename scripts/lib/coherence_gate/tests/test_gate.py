"""The paired gate: tier 0 identity, tier 1 graders + paired degeneracy, tier 2 judge routing,
fake-token refusal, the INCOMPLETE path, the CLI. No inference.

Includes the 10 verdict tests of gpu-block-27b-20261003/test_q38_t7_verdict.py, re-expressed on the
library (q38_t7's drafted/nodraft arms are the library's cand/base)."""
from __future__ import annotations

import hashlib
import json

import pytest
from conftest import TOK, enc, needs_tokenizer

import coherence_gate as CG
from coherence_gate import answers as A
from coherence_gate import degeneracy as D
from coherence_gate.__main__ import main as cli_main

SENT = ("The kernel reads each weight row once per token, so decode is bound by memory bandwidth "
        "rather than arithmetic; the fix is to keep the activations resident in cache. ")
GOOD = SENT * 1 + "Then it checks the result against the reference and reports the difference."
OTHER = "A different but fluent answer: prefill is compute bound, decode is bandwidth bound on this host."
LOOP = "Wait, let me check again. " * 60
CORRECT_MC = {"id": "mc", "expected": "E", "scoring_method": "multiple_choice"}


def row(pid, text, **kw):
    return {"id": pid, "text": text, **kw}


def ev(base, cand, truth=None, **kw):
    kw.setdefault("allow_text_only", True)
    return CG.evaluate(base, cand, truth, **kw)


def item(rep, pid=None):
    its = rep["items"]
    return its[0] if pid is None else next(i for i in its if i["id"] == pid)


# =========================================================================== tier 0
def test_identical_outputs_pass_at_tier0():
    rep = ev([row("p", GOOD)], [row("p", GOOD)])
    it = item(rep)
    assert (it["tier"], it["verdict"]) == (0, "PASS") and it["identity"]["identical"]
    assert rep["aggregate"]["gate"] == "PASS" and rep["aggregate"]["identical"] == 1


def test_identity_uses_token_ids_when_both_arms_carry_them():
    ids = [101, 7, 9, 4000, 12]
    rep = CG.evaluate([row("p", GOOD, token_ids=ids)], [row("p", GOOD, token_ids=ids[:-1] + [13])])
    it = item(rep)
    assert not it["identity"]["identical"] and it["identity"]["basis"] == "sha256(text)+token_ids"


def test_sha256_is_verified_never_trusted_alone():
    good = hashlib.sha256(GOOD.encode()).hexdigest()
    rep = ev([row("p", GOOD, sha256=good)], [row("p", GOOD)])
    assert item(rep)["identity"]["base_sha256"] == good
    with pytest.raises(CG.GateInputError, match="does not match"):
        ev([row("p", GOOD, sha256="0" * 64)], [row("p", GOOD)])
    with pytest.raises(CG.GateInputError, match="no output text"):
        ev([{"id": "p", "sha256": good}], [row("p", GOOD)])


def test_identical_but_degenerate_is_pass_with_shared_defect_reported():
    rep = ev([row("p", LOOP)], [row("p", LOOP)])
    it = item(rep)
    assert it["verdict"] == "PASS" and any("shared defect" in r for r in it["reasons"])
    assert rep["aggregate"]["identical_with_shared_defect"] == 1


def test_full_texts_are_stored_inline_or_by_path(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text(OTHER)
    rep = ev([row("p", GOOD)], [{"id": "p", "text_path": str(p)}])
    t = item(rep)["texts"]
    assert t["base"]["text"] == GOOD and t["cand"]["path"] == str(p)
    assert t["cand"]["sha256"] == hashlib.sha256(OTHER.encode()).hexdigest()


# =========================================================================== tier 1 paired logic
def test_drafted_worse_than_nodraft_is_regression():                      # q38 verdict test
    rep = ev([row("p", GOOD)], [row("p", LOOP)])
    it = item(rep)
    assert (it["tier"], it["verdict"]) == (1, "REGRESSION") and "degeneracy OK -> DEGENERATE" in it["reasons"][0]
    assert rep["aggregate"]["gate"] == "FAIL"


def test_shared_nonok_is_both_bad_not_failure():                         # q38 verdict test
    rep = ev([row("gsm", LOOP), row("mc", "E")], [row("gsm", LOOP + " x"), row("mc", "E")],
             [{"id": "mc", "expected": "E", "scoring_method": "multiple_choice"}])
    assert item(rep, "gsm")["verdict"] == "BOTH_BAD" and item(rep, "mc")["verdict"] == "PASS"
    assert rep["aggregate"]["gate"] == "PASS" and rep["aggregate"]["both_bad"] == 1


def test_answer_correct_to_wrong_fails_even_when_both_ok():               # q38 verdict test
    rep = ev([row("mc", "<answer>E</answer>")], [row("mc", "<answer>A</answer>")], [CORRECT_MC])
    it = item(rep)
    assert it["verdict"] == "REGRESSION" and "answer correct -> wrong" in it["reasons"][0]


def test_answer_correct_to_unanswered_is_regression():
    rep = ev([row("q", "so <answer>98</answer>")], [row("q", "1. **Determine the cost of a half")],
             [{"id": "q", "expected": "98", "grader": "gsm8k"}])
    assert item(rep)["verdict"] == "REGRESSION"


def test_both_wrong_is_model_not_drafting():                              # q38 verdict test
    t = [{"id": "qa", "expected": "Romanian", "scoring_method": "f1"}]
    rep = ev([row("qa", "<answer>English</answer>")], [row("qa", "<answer>English.</answer>")], t)
    assert item(rep)["verdict"] == "BOTH_BAD" and rep["aggregate"]["gate"] == "PASS"


def test_http_error_in_candidate_is_regression_and_in_base_needs_review():  # q38 verdict test
    rep = ev([row("p", GOOD)], [row("p", "", http_ok=False)])
    assert item(rep)["verdict"] == "REGRESSION" and rep["aggregate"]["gate"] == "FAIL"
    rep = ev([row("p", "", error="timeout")], [row("p", GOOD)])
    assert item(rep)["verdict"] == "NEEDS_REVIEW" and rep["aggregate"]["gate"] == "INCOMPLETE"
    rep = ev([row("p", "", http=500)], [row("p", "", http=502)])
    assert item(rep)["verdict"] == "BOTH_BAD"


def test_diverged_correct_answer_passes_at_tier1_and_improvement_is_pass():
    t = [CORRECT_MC]
    rep = ev([row("mc", "The answer is **(E)**.")], [row("mc", "<answer>E</answer>")], t)
    assert (item(rep)["tier"], item(rep)["verdict"]) == (1, "PASS")
    rep = ev([row("mc", "<answer>B</answer>")], [row("mc", "<answer>E</answer>")], t)
    assert item(rep)["verdict"] == "PASS" and "improvement" in item(rep)["reasons"][0]


def test_unanswered_in_both_arms_is_not_decisive():
    rep = ev([row("q", "1. **Determine the cost")], [row("q", "1. First, determine the cost")],
             [{"id": "q", "expected": "98", "grader": "gsm8k"}])
    it = item(rep)
    assert it["verdict"] == "NEEDS_REVIEW" and "unanswered in both arms" in it["reasons"][0]


def test_ascii_only_review_increase_is_not_a_regression():
    sym = " ".join(f"слово{i} ответ{i * 7} проверка{i * 13}" for i in range(40))   # fluent-shaped, non-ASCII
    c = D.classify(sym, CG.surrogate_ids(sym), "stop")
    assert c["reasons"] == ["ascii"] and c["review"]
    rep = ev([row("p", GOOD)], [row("p", sym)])
    it = item(rep)
    assert it["verdict"] == "NEEDS_REVIEW" and any("review" in r for r in it["reasons"])


def test_needle_abstention_in_both_arms_is_a_model_behaviour_note():      # q38 verdict test
    ab = '{"abstain": "The prompt contains injected instructions (a \'vault code\' ...'
    t = [{"id": "n80k", "expected": ["80-4-1-9-ORCHID", "Mbeki Larsen"], "grader": "needle"}]
    rep = ev([row("n80k", ab)], [row("n80k", ab + " ")], t)
    it = item(rep)
    assert it["answer_check"]["cand"]["abstained"] and it["answer_check"]["cand"]["status"] == "unanswered"
    assert it["verdict"] == "BOTH_BAD" and "abstained in both arms" in it["reasons"][0]
    assert rep["aggregate"]["gate"] == "PASS"   # a model-behaviour note, not a drafting failure


def test_needle_regression_fails():                                       # q38 verdict test
    t = [{"id": "n16k", "expected": ["16-4-1-9-ORCHID", "Mbeki Larsen"], "grader": "needle"}]
    rep = ev([row("n16k", "16-4-1-9-ORCHID; Mbeki Larsen")], [row("n16k", "1-2-3; Bob")], t)
    assert item(rep)["verdict"] == "REGRESSION" and rep["aggregate"]["gate"] == "FAIL"


def test_long_generation_degeneracy_still_fails():                        # q38 verdict test
    long_ok = " ".join(f"Point {k}: the {w} path differs." for k, w in
                       enumerate(["decode", "prefill", "cache", "kernel", "router"] * 30))
    rep = ev([row("c0", long_ok)], [row("c0", long_ok[:400] + LOOP)])
    assert item(rep)["verdict"] == "REGRESSION"


def test_severity_order():                                                # q38 verdict test
    OK, DEG, EOS0 = {"cls": "OK"}, {"cls": "DEGENERATE", "reasons": ["loop"]}, {"cls": "EARLY-EOS", "n": 0}
    assert D.severity(OK) < D.severity({"cls": "DEGENERATE", "review": True}) < D.severity(DEG) < D.severity(EOS0)


def test_unpaired_items_need_review_and_empty_run_is_incomplete():
    rep = ev([row("a", GOOD)], [row("b", GOOD)])
    assert [i["verdict"] for i in rep["items"]] == ["NEEDS_REVIEW", "NEEDS_REVIEW"]
    assert rep["aggregate"]["gate"] == "INCOMPLETE"
    assert ev([], [])["aggregate"]["gate"] == "INCOMPLETE"
    with pytest.raises(CG.GateInputError, match="duplicate"):
        ev([row("a", GOOD), row("a", GOOD)], [])


# =========================================================================== tier 2 / INCOMPLETE
def test_unjudged_divergence_makes_the_gate_incomplete():
    rep = ev([row("p", GOOD)], [row("p", OTHER)])
    it = item(rep)
    assert (it["tier"], it["verdict"]) == (1, "NEEDS_REVIEW") and "unjudged" in it["reasons"][-1]
    agg = rep["aggregate"]
    assert agg["gate"] == "INCOMPLETE" and "no judge was given" in agg["gate_reasons"][0]


def test_judge_is_called_only_for_diverged_items_without_ground_truth():
    calls = []

    def judge(pair):
        calls.append(pair["id"])
        assert pair["base"]["text"] and pair["cand"]["text"]
        return {"verdict": "PASS", "reasons": ["equivalent"]}
    judge.judge_id = "test-judge.v0"
    base = [row("same", GOOD), row("mc", "<answer>E</answer>"), row("open", GOOD), row("loop", GOOD)]
    cand = [row("same", GOOD), row("mc", "The answer is E."), row("open", OTHER), row("loop", LOOP)]
    rep = ev(base, cand, [CORRECT_MC | {"id": "mc"}], judge_fn=judge)
    assert calls == ["open"]
    assert (item(rep, "open")["tier"], item(rep, "open")["verdict"]) == (2, "PASS")
    assert item(rep, "open")["judge"]["id"] == "test-judge.v0" and rep["judge"] == "test-judge.v0"
    assert rep["aggregate"]["gate"] == "FAIL"          # the loop regression still decides


def test_judge_failure_or_garbage_is_needs_review():
    def boom(pair):
        raise RuntimeError("endpoint down")
    rep = ev([row("p", GOOD)], [row("p", OTHER)], judge_fn=boom)
    assert item(rep)["verdict"] == "NEEDS_REVIEW" and "judge error" in item(rep)["reasons"][-1]
    rep = ev([row("p", GOOD)], [row("p", OTHER)], judge_fn=lambda p: "LOOKS_FINE")
    assert item(rep)["verdict"] == "NEEDS_REVIEW" and rep["aggregate"]["gate"] == "INCOMPLETE"
    rep = ev([row("p", GOOD)], [row("p", OTHER)], judge_fn=lambda p: "REGRESSION")
    assert rep["aggregate"]["gate"] == "FAIL"


# =========================================================================== token provenance
def test_inf70_v1_fake_token_pattern_is_refused():
    """Regression: every INF-70 chat client called classify(dict(tokens=list(range(npred)), ...)),
    which made uniq/top/run vacuous. The library refuses those ids, even with a tokenizer available."""
    npred = 200
    text = (SENT * 20)[:1200]
    with pytest.raises(CG.TokenProvenanceError, match="list\\(range"):
        CG.evaluate([row("p", text, token_ids=list(range(npred)))], [row("p", text + ".")])
    tok = CG.FnTokenizer(CG.surrogate_ids, "fn:surrogate-as-tokenizer")
    with pytest.raises(CG.TokenProvenanceError):
        CG.evaluate([row("p", text, tokens=list(range(npred)))], [row("p", text)], tokenizer=tok)


def test_fake_id_shapes():
    assert CG.fake_reason(list(range(4)))                       # the literal placeholder
    assert CG.fake_reason(list(range(5000, 5016)))               # any offset, long run
    assert CG.fake_reason([1, 2, "3"]) and CG.fake_reason([-1, 5])
    assert CG.fake_reason([7] * 50, "short") and "do not belong" in CG.fake_reason([7] * 50, "short")
    assert CG.fake_reason([16, 17, 18, 19], "1234") is None     # Qwen digits are consecutive ids: real
    assert CG.fake_reason([0, 1, 2], "!\"#") is None             # below FAKE_MIN_N


def test_no_token_source_is_refused_unless_text_only_is_declared():
    with pytest.raises(CG.TokenProvenanceError, match="not declared"):
        CG.evaluate([row("p", GOOD)], [row("p", OTHER)])
    rep = CG.evaluate([row("p", GOOD)], [row("p", OTHER)], allow_text_only=True)
    prov = item(rep)["degeneracy"]["token_provenance"]
    assert prov["base"] == prov["cand"] == {"mode": "text_only", "source": CG.tokens.SURROGATE_ID, "calibrated": False}
    assert rep["text_only_declared"] is True


def test_text_only_is_symmetric_when_one_arm_has_ids():
    rep = CG.evaluate([row("p", GOOD, token_ids=[5, 9, 11, 3])], [row("p", OTHER)], allow_text_only=True)
    prov = item(rep)["degeneracy"]["token_provenance"]
    assert prov["base"]["mode"] == prov["cand"]["mode"] == "text_only"


@needs_tokenizer
def test_tokenizer_fallback_records_its_id_and_real_ids_win():
    tok = CG.FnTokenizer(enc, "hf:qwen3.8-27b-test")
    rep = CG.evaluate([row("p", GOOD, token_ids=enc(GOOD))], [row("p", OTHER)], tokenizer=tok)
    prov = item(rep)["degeneracy"]["token_provenance"]
    assert prov["base"]["mode"] == "token_ids" and prov["cand"] == {
        "mode": "tokenizer", "source": "hf:qwen3.8-27b-test", "calibrated": True}
    assert rep["tokenizer"] == "hf:qwen3.8-27b-test"
    assert TOK is not None


# =========================================================================== graders
def test_grade_against_question_pool_shapes():                            # q38 verdict test
    mc = {"expected": "E", "scoring_method": "multiple_choice"}
    assert A.grade("E", mc)["status"] == "correct" and A.grade("B", mc)["status"] == "wrong"
    assert A.grade("The answer is **(E)**.", mc)["status"] == "correct"
    assert A.grade("1. **Determine the cost", mc)["status"] == "unanswered"
    em = {"id": "gsm8k_00001", "expected": "98", "scoring_method": "exact_match",
          "scoring_config": {"extract_pattern": "<answer>(.*?)</answer>"}}
    assert A.grader_for(em) == "gsm8k"
    assert A.grade("... so <answer>$98</answer>", em)["status"] == "correct"
    assert A.grade("1. **Determine the cost of a half-gallon jar", em)["status"] == "unanswered"
    f1 = {"expected": "Romanian", "scoring_method": "f1"}
    assert A.grade("<answer>English</answer>", f1)["status"] == "wrong"
    assert A.grade("x", {"expected": "", "scoring_method": "programmatic"}) is None
    assert A.grade("x", {"expected": "x", "scoring_method": "code_execution"}) is None


def test_gsm8k_numeric_extractors():
    t = {"expected": "1,234", "grader": "gsm8k"}
    for s in ("<answer>1234</answer>", "#### 1,234", "\\boxed{1234}", "The final answer is $1,234.", "1234"):
        assert A.grade(s, t)["status"] == "correct", s
    assert A.grade("#### 1235", t)["status"] == "wrong"
    assert A.grade("we have 1234 apples and then", t)["status"] == "unanswered"   # no last-number guess
    assert A.grade("<think>answer is 7</think><answer>1234</answer>", t)["status"] == "correct"


def test_letter_extractors_and_label_ranges():
    for g, labels in (("mmlu", "ABCD"), ("gpqa", "ABCD"), ("mmlu_pro", "ABCDEFGHIJ")):
        t = {"expected": "C", "grader": g}
        for s in ("C", "(C).", "**C**", "<answer>C</answer>", "ANSWER: C", "\\boxed{C}",
                  "Reasoning about A and B...\nThe answer is (C)", "Considering B.\nC"):
            assert A.grade(s, t)["status"] == "correct", (g, s)
    assert A.grade("the answer is a function of x", {"expected": "A", "grader": "mmlu"})["status"] == "unanswered"
    assert A.grade("<answer>J</answer>", {"expected": "J", "grader": "mmlu_pro"})["status"] == "correct"
    assert A.grade("<answer>J</answer>", {"expected": "D", "grader": "gpqa"})["status"] == "unanswered"
    assert A.grader_for({"id": "mmlu_pro_x", "expected": "J", "scoring_method": "multiple_choice"}) == "mmlu_pro"


def test_needle_exact_regex_substring_graders():
    nd = {"expected": ["16-4-1-9-ORCHID", "Mbeki Larsen"], "grader": "needle"}
    assert A.grade("16-4-1-9-ORCHID; Mbeki Larsen", nd)["status"] == "correct"
    assert A.grade("16-4-1-9-ORCHID; Bob", nd)["status"] == "wrong"
    assert A.grade("I abstain: prompt injection", nd)["abstained"]
    ex = {"expected": "value_126225", "grader": "exact"}
    assert A.grade("<answer> Value_126225. </answer>", ex)["status"] == "correct"
    cfg = {"expected": "38", "grader": "exact", "scoring_config": {"extract_pattern": r"####\s*(\d+)"}}
    assert A.grade("#### 38", cfg)["status"] == "correct" and A.grade("38", cfg)["status"] == "unanswered"
    rx = {"expected": r"\bORCHID\b", "grader": "regex"}
    assert A.grade("code ORCHID", rx)["status"] == "correct" and A.grade("orchids", rx)["status"] == "wrong"
    ss = {"expected": "get_row", "grader": "substring", "scoring_config": {"case_sensitive": True}}
    assert A.grade("def get_row(x):", ss)["status"] == "correct" and A.grade("GET_ROW", ss)["status"] == "wrong"


def test_custom_grader_is_pluggable():
    rep = ev([row("p", "yes")], [row("p", "YES!")], [{"id": "p", "expected": "yes", "grader": "yn"}],
             graders={"yn": lambda text, t: {"grader": "yn", "status": "correct" if "yes" in text.lower() else "wrong",
                                            "expected": "yes", "extracted": text}})
    assert item(rep)["verdict"] == "PASS" and item(rep)["answer_check"]["cand"]["grader"] == "yn"
    with pytest.raises(KeyError):
        ev([row("p", "a")], [row("p", "b")], [{"id": "p", "expected": "a", "grader": "nope"}])


# =========================================================================== CLI + schema
def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_cli_exit_codes_and_schema(tmp_path, capsys):
    b = _jsonl(tmp_path / "b.jsonl", [row("p", GOOD), row("mc", "<answer>E</answer>")])
    t = _jsonl(tmp_path / "t.jsonl", [CORRECT_MC | {"id": "mc"}])
    same = _jsonl(tmp_path / "same.jsonl", [row("p", GOOD), row("mc", "E")])
    worse = _jsonl(tmp_path / "worse.jsonl", [row("p", GOOD), row("mc", "<answer>A</answer>")])
    open_ = _jsonl(tmp_path / "open.jsonl", [row("p", OTHER), row("mc", "E")])
    fake = _jsonl(tmp_path / "fake.jsonl", [row("p", GOOD, token_ids=list(range(40))), row("mc", "E")])
    out = tmp_path / "v.json"
    assert cli_main(["--base", b, "--cand", same, "--truth", t, "--text-only", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["schema"] == CG.SCHEMA_ID and rec["degeneracy_classifier"] == "degeneracy.v2"
    assert set(rec["aggregate"]) >= {"n", "identical", "regressions", "both_bad", "needs_review", "gate"}
    for it in rec["items"]:
        assert set(it) >= {"tier", "verdict", "reasons", "degeneracy", "answer_check", "texts"}
        assert it["verdict"] in CG.VERDICTS
    assert rec["inputs"]["base"]["sha256"]
    assert cli_main(["--base", b, "--cand", worse, "--truth", t, "--text-only", "--summary"]) == 1
    assert cli_main(["--base", b, "--cand", open_, "--truth", t, "--text-only", "--summary"]) == 2
    assert cli_main(["--base", b, "--cand", same, "--summary"]) == 3          # no token source declared
    assert cli_main(["--base", b, "--cand", fake, "--text-only", "--summary"]) == 3
    assert "REFUSED" in capsys.readouterr().err


def test_load_jsonl_keeps_unicode_line_separators_inside_strings(tmp_path):
    """question_pool.jsonl carries U+2028 inside strings; str.splitlines() would split the row."""
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"id": "x", "expected": "a b"}, ensure_ascii=False) + "\n")
    assert CG.load_jsonl(p) == [{"id": "x", "expected": "a b"}]


def test_question_pool_substring_is_not_auto_mapped_to_ground_truth():
    """On code suites question_pool's `substring` is a presence check ("def ", a function name)."""
    t = {"id": "mbpp_0303", "expected": "def ", "scoring_method": "substring"}
    assert A.grader_for(t) is None and A.grader_for(t | {"grader": "substring"}) == "substring"


def test_text_only_surrogate_splits_digit_loops_and_clusters_punctuation():
    ids = CG.surrogate_ids("2" * 60)
    assert len(ids) == 20 and len(set(ids)) == 1               # a `2222...` loop stays visible
    # x = f ( a )): ** bold ** ```  -> 10 (a per-character split would give 16)
    assert len(CG.surrogate_ids("x = f(a)):  **bold** ```")) == 10


def test_anchor_is_recorded_and_validated(tmp_path):
    a = {"source_commit": "ffc1bac82", "binary_sha256": "ab" * 32, "linkage_sha256": "cd" * 32}
    rep = ev([row("p", GOOD)], [row("p", GOOD)], anchor=a)
    assert rep["anchor"] == a and rep["anchor_named"] is True
    assert ev([row("p", GOOD)], [row("p", GOOD)])["anchor_named"] is False
    with pytest.raises(CG.GateInputError, match="anchor"):
        ev([row("p", GOOD)], [row("p", GOOD)], anchor={"binary_sha256": ""})
    b = _jsonl(tmp_path / "b.jsonl", [row("p", GOOD)])
    out = tmp_path / "v.json"
    assert cli_main(["--base", b, "--cand", b, "--text-only", "--anchor-json", json.dumps(a), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["anchor"] == a
    assert cli_main(["--base", b, "--cand", b, "--text-only", "--anchor-json", "{not json", "--summary"]) == 3
