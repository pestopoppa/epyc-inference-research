"""Conformance fixtures for stt_norm (CS-8.STT1 (a)/(b), DP-B). Synthetic strings only."""
from __future__ import annotations

import inspect

import pytest

from scripts.kernel_rnd.autokernel.adapters import whisper_stt
from scripts.speech.stt_bakeoff import stt_norm as N

REF, HYP = N.REFERENCE, N.HYPOTHESIS

CORPUS = [
    ("Hello, World! [BLANK_AUDIO] don't stop", "en"),
    ("In 1920 there were 21 men and 3 dogs", "en"),
    ("STRAẞE ﬁne Kelvin K", "en"),
    ("a,b c-d e.f", "en"),
    ("[ Silence ] (silence) * [MUSIC] the end", "en"),
    ("l’uomo dell'anno, c'est la vie", "it"),
    ("uh um euh ähm <unk> word", "de"),
    ("", "en"),
    ("   spaced    out   ", "en"),
]


# --- signature: the normaliser can never read the partner string -------------

@pytest.mark.parametrize("fn", [N.normalise_v1, N.normalise_v2_candidate,
                                N.normalise_secondary, N.V1.normalise])
def test_signature_is_text_lang_side_only(fn):
    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    assert params == ["text", "lang", "side"]
    assert inspect.signature(fn).parameters["side"].kind is inspect.Parameter.KEYWORD_ONLY


def test_passing_reference_to_normaliser_is_a_type_error():
    with pytest.raises(TypeError):
        N.normalise_v1("hyp", "en", side=HYP, reference="ref")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        N.normalise_v1("hyp", "en", "ref")  # type: ignore[misc]


def test_no_partner_aware_entry_point_exists():
    names = [n for n in dir(N) if callable(getattr(N, n))]
    for name in names:
        try:
            sig = inspect.signature(getattr(N, name))
        except (TypeError, ValueError):
            continue
        for p in sig.parameters:
            assert p not in ("reference", "partner", "ref_text", "hypothesis_rewrite")


def test_side_is_validated():
    with pytest.raises(ValueError):
        N.normalise_v1("x", "en", side="both")


# --- §1.3 properties -----------------------------------------------------------

@pytest.mark.parametrize("nid", sorted(N.NORMALISERS))
def test_properties_pass_on_corpus(nid):
    rep = N.check_properties(CORPUS, nid)
    assert rep.check.outcome == whisper_stt.schemas.PASS, rep.violations
    assert (rep.symmetric, rep.idempotent, rep.deterministic) == (True, True, True)


@pytest.mark.parametrize("text,lang", CORPUS)
@pytest.mark.parametrize("side", [REF, HYP])
def test_idempotence(text, lang, side):
    once = N.normalise_v1(text, lang, side=side)
    assert N.normalise_v1(once.text, lang, side=side).text == once.text


def test_symmetry_outside_numeral_step():
    for text, lang in CORPUS:
        if any(c.isdigit() for c in text):
            continue
        r = N.normalise_v1(text, lang, side=REF)
        h = N.normalise_v1(text, lang, side=HYP)
        assert (r.tokens, r.markers_removed_count, r.unknown_markers, r.unk_count) == (
            h.tokens, h.markers_removed_count, h.unknown_markers, h.unk_count)


def test_determinism_repeated_calls_equal():
    a = [N.normalise_v2_candidate(t, l, side=HYP) for t, l in CORPUS]
    b = [N.normalise_v2_candidate(t, l, side=HYP) for t, l in CORPUS]
    assert a == b


def test_empty_corpus_is_could_not_check_not_pass():
    rep = N.check_properties([], N.NORMALISER_ID)
    assert rep.check.outcome == whisper_stt.schemas.COULD_NOT_CHECK


# --- pipeline steps -------------------------------------------------------------

def test_nfkc_then_casefold_not_lower():
    r = N.normalise_v1("STRAẞE ﬁne K", "en", side=REF)
    assert r.tokens == ("strasse", "fine", "k")


def test_closed_markers_removed_and_counted():
    r = N.normalise_v1("[BLANK_AUDIO] hi [ Silence ] (silence) * [Music]", "en", side=HYP)
    assert r.tokens == ("hi",)
    assert r.markers_removed_count == 5
    assert r.status == N.STATUS_OK


def test_unknown_marker_not_removed_and_routed():
    r = N.normalise_v1("yes [applause] (inaudible)", "en", side=HYP)
    assert r.unknown_markers == ("[applause]", "(inaudible)")
    assert r.status == N.STATUS_UNKNOWN_MARKER
    assert "applause" in r.tokens  # kept, never deleted


def test_unk_token_is_counted_not_a_marker():
    r = N.normalise_v1("the <unk> cat <UNK>", "en", side=HYP)
    assert r.unk_count == 2
    assert r.unknown_markers == ()
    assert r.tokens == ("the", "unk", "cat", "unk")


def test_punctuation_to_separator_never_deletion():
    assert N.normalise_v1("a,b", "en", side=REF).tokens == ("a", "b")
    assert N.normalise_v1("large-v3x", "en", side=REF).tokens == ("large", "v3x")


def test_apostrophes_kept():
    assert N.normalise_v1("Don't", "en", side=REF).tokens == ("don't",)


def test_whitespace_collapsed_last():
    assert N.normalise_v1("  a \t\n b  ", "en", side=REF).text == "a b"


def test_v1_has_no_filler_step():
    r = N.normalise_v1("uh um yes", "en", side=REF)
    assert r.tokens == ("uh", "um", "yes")
    assert "remove_enumerated_fillers" not in N.V1_STEPS


# --- numerals: hypothesis side only, closed table, COULD_NOT_CHECK ----------------

def test_hypothesis_numerals_converted_from_closed_table():
    r = N.normalise_v1("in 1920 the 21st of 100", "en", side=HYP)
    assert r.text == "in nineteen twenty the twenty first of one hundred"
    assert r.numerals_converted == 3 and r.status == N.STATUS_OK


def test_reference_side_never_converted_and_flagged():
    r = N.normalise_v1("in 1920", "en", side=REF)
    assert r.tokens == ("in", "1920")
    assert r.status == N.STATUS_NUMERAL_UNCOVERED


@pytest.mark.parametrize("text,offender", [
    ("year 1905", "1905"), ("in 2024", "2024"), ("costs $5", "$5"),
    ("pi is 3.14", "3.14"), ("50% off", "50%"), ("model v3", "v3"),
    ("1,920 people", "1,920")])
def test_uncovered_numeral_is_could_not_check(text, offender):
    r = N.normalise_v1(text, "en", side=HYP)
    assert r.status == N.STATUS_NUMERAL_UNCOVERED
    assert offender in r.numeral_uncovered


def test_v1_has_no_non_english_table():
    assert N.normalise_v1("21", "it", side=HYP).status == N.STATUS_NUMERAL_UNCOVERED


def test_table_is_closed_and_enumerated():
    assert set(N.V1_NUMERAL_TABLES) == {"en"}
    assert "101" not in N.NUMERAL_TABLE_EN and "1905" not in N.NUMERAL_TABLE_EN
    assert N.NUMERAL_TABLE_EN["2000"] == "two thousand"


# --- v2 candidate --------------------------------------------------------------

@pytest.mark.parametrize("lang,digits,spelled", [
    ("it", "21 28 23 100", "ventuno ventotto ventitré cento"),
    ("de", "21 30 100", "einundzwanzig dreissig hundert"),
    ("fr", "21 71 80 91", "vingt et un soixante et onze quatre vingts quatre vingt onze"),
])
def test_v2_language_tables(lang, digits, spelled):
    assert N.normalise_v2_candidate(digits, lang, side=HYP).text == spelled


def test_v2_typographic_apostrophe_elision_one_token():
    assert N.normalise_v1("l’uomo", "it", side=REF).tokens == ("l", "uomo")
    assert N.normalise_v2_candidate("l’uomo", "it", side=REF).tokens == ("l'uomo",)
    assert N.normalise_v2_candidate("dell'anno", "it", side=REF).tokens == ("dell'anno",)


def test_v2_primary_keeps_fillers_dp_b():
    r = N.normalise_v2_candidate("euh oui euh", "fr", side=REF)
    assert r.tokens == ("euh", "oui", "euh") and r.fillers_removed_count == 0
    assert N.count_fillers(r.tokens, "fr") == 2


def test_secondary_removes_closed_fillers_counted():
    r = N.normalise_secondary("ähm ja äh genau", "de", side=HYP)
    assert r.tokens == ("ja", "genau") and r.fillers_removed_count == 2
    assert N.SECONDARY.secondary and N.SECONDARY.column == "secondary"


def test_v2_does_not_merge_german_compounds():
    assert N.normalise_v2_candidate("Bahn Hof", "de", side=HYP).tokens == ("bahn", "hof")


# --- identity and contract -------------------------------------------------------

def test_hash_is_deterministic_and_distinct_per_id():
    hs = {nid: N.normaliser_sha256(nid) for nid in N.NORMALISERS}
    assert len(set(hs.values())) == 3
    assert all(len(h) == 64 for h in hs.values())
    N.normaliser_sha256.cache_clear()
    assert N.normaliser_sha256(N.NORMALISER_ID) == hs[N.NORMALISER_ID]


def test_v1_passes_whisper_contract_and_candidates_do_not():
    assert N.contract_check(N.NORMALISER_ID).outcome == whisper_stt.schemas.PASS
    assert N.V1_STEPS == whisper_stt.NORMALIZATION_STEPS
    assert N.contract_check(N.V2_CANDIDATE_ID).outcome == whisper_stt.schemas.FAIL
    assert N.contract_check(N.SECONDARY_ID).outcome == whisper_stt.schemas.FAIL
    assert N.V2_CANDIDATE.candidate and not N.V1.candidate


def test_marker_list_is_reused_from_whisper_stt():
    assert len(N.MARKERS_CASEFOLDED) == len(set(
        m.casefold() for m in whisper_stt.NONLEXICAL_MARKERS))
