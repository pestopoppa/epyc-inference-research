"""Conformance fixtures for the STT bake-off scorer (CS-8.STT1 (c)-(h)).

Synthetic, hand-made strings only — no corpus audio or real reference text.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json

import pytest

from scripts.kernel_rnd.autokernel.adapters import whisper_stt
from scripts.speech.stt_bakeoff import scorer as S
from scripts.speech.stt_bakeoff import stt_norm

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra").split()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def row(uid, ref, hyp=None, **kw):
    r = {"utterance_id": uid, "pcm_sha256": sha("pcm:" + uid), "reference_raw": ref,
         "language": "en", "set_id": "setA", "reference_track": "standard",
         "content_stratum": "clean_read"}
    if hyp is not None:
        r["hypothesis_raw"] = hyp
    r.update(kw)
    return r


def manifest(rows, *, arm=None, stratum="public", segmentation_mode="none", **kw):
    m = {"schema_version": S.RAW_MANIFEST_SCHEMA_VERSION, "corpus_id": "synth",
         "stratum": stratum, "language_mode": "forced",
         "segmentation_mode": segmentation_mode,
         "arm": arm or {"arm_id": "armA", "arm_kind": S.ARM_KIND_OFFLINE,
                        "runner_id": "runner-x"},
         "rows": rows}
    m.update(kw)
    return m


def score(m, column="v1", **kw):
    return S.score_arm(m, column=column, bootstrap_resamples=kw.pop("resamples", 200),
                       bootstrap_seed=kw.pop("seed", 42), **kw)


def per_utt(result, uid):
    return next(u for u in result["per_utterance"] if u["utterance_id"] == uid)


def known_answer_rows(errors_in_first=21):
    """100 utterances, 1870 reference tokens (70x19 + 30x18), 63 substitutions."""
    rows = []
    for i in range(100):
        ref = list(WORDS[: 19 if i < 70 else 18])
        hyp = list(ref)
        if i < errors_in_first:
            hyp[0:3] = ["xray", "yankee", "zulu"]
        rows.append(row(f"u{i:03d}", " ".join(ref), " ".join(hyp),
                        set_id="setA" if i < 50 else "setB",
                        content_stratum="noise" if i % 2 == 0 else "clean_read"))
    return rows


# --- Annex S Appendix S-A known answer -------------------------------------------

@pytest.fixture(scope="module")
def known():
    return S.score_arm(manifest(known_answer_rows()), column="v1",
                       bootstrap_resamples=2000, bootstrap_seed=42)


def test_known_answer_pooled_wer(known):
    p = known["records"][0]
    assert (p["errors"], p["reference_tokens"], p["n"]) == (63, 1870, 100)
    assert p["wer"] == 100.0 * 63 / 1870
    assert p["wer"] == whisper_stt.pooled_corpus_wer(
        [u["errors"] for u in known["per_utterance"]],
        [u["reference_tokens"] for u in known["per_utterance"]])


def test_known_answer_bootstrap_pinned_and_deterministic(known):
    p = known["records"][0]
    assert p["bootstrap"]["resamples"] == 2000 and p["bootstrap"]["seed"] == 42
    errs = [u["errors"] for u in known["per_utterance"]]
    refs = [u["reference_tokens"] for u in known["per_utterance"]]
    a = S.bootstrap_pooled_wer(errs, refs, resamples=2000, seed=42)
    b = S.bootstrap_pooled_wer(errs, refs, resamples=2000, seed=42)
    assert a == b == tuple(p["ci95"])
    assert a[0] < p["wer"] < a[1]
    # Pinned procedure (random.Random(42), percentile, linear interpolation).
    assert a == pytest.approx((2.092274678111588, 4.657387580299786), abs=1e-12)
    assert S.bootstrap_pooled_wer(errs, refs, resamples=2000, seed=43) != a


def test_paired_identical_arms_is_zero(known):
    out = S.paired_bootstrap(known, known, resamples=2000, seed=42)
    assert out["point_delta_pp"] == 0.0 and out["interval_pp"] == [0.0, 0.0]
    assert out["n_paired"] == 100 and out["n_differing_utterances"] == 0


def test_paired_delta_repools_numerator_and_denominator(known):
    rows = known_answer_rows(errors_in_first=22)  # one more utterance with 3 errors
    other = S.score_arm(manifest(rows, arm={"arm_id": "armB", "arm_kind": S.ARM_KIND_OFFLINE,
                                            "runner_id": "r"}),
                        column="v1", bootstrap_resamples=2000, bootstrap_seed=42)
    out = S.paired_bootstrap(known, other, resamples=4000, seed=42)
    assert out["point_delta_pp"] == pytest.approx(100.0 * 3 / 1870)
    assert out["n_differing_utterances"] == 1
    lo, hi = out["interval_pp"]
    assert lo == 0.0 and hi > 0.0
    assert out == S.paired_bootstrap(known, other, resamples=4000, seed=42)


def test_paired_refuses_different_audio(known):
    other = copy.deepcopy(known)
    other["per_utterance"][5]["pcm_sha256"] = sha("different audio")
    with pytest.raises(S.CorpusMismatch):
        S.paired_bootstrap(known, other, resamples=10, seed=1)


def test_bootstrap_requires_explicit_seed_and_count():
    sig = inspect.signature(S.bootstrap_pooled_wer)
    assert sig.parameters["resamples"].default is inspect.Parameter.empty
    assert sig.parameters["seed"].default is inspect.Parameter.empty
    with pytest.raises(S.ScorerError):
        S.bootstrap_pooled_wer([1], [5], resamples=0, seed=1)


# --- (e) per-set / unweighted mean / pooled / per content stratum ------------------

def test_set_and_stratum_wers_side_by_side_and_labelled(known):
    p = known["records"][0]
    assert p["wer_kind"] == "pooled_corpus_wer"
    sets = {r["scope_id"]: r for r in known["records"] if r["scope"] == "set"}
    assert sets["setA"]["wer"] == 100.0 * 63 / (50 * 19)
    assert sets["setB"]["wer"] == 0.0
    assert p["set_wers"] == {"setA": sets["setA"]["wer"], "setB": 0.0}
    assert p["unweighted_mean_of_set_wers"] == pytest.approx(sets["setA"]["wer"] / 2)
    assert p["unweighted_mean_of_set_wers"] != p["wer"]
    strata = {r["scope_id"] for r in known["records"] if r["scope"] == "content_stratum"}
    assert strata == {"noise", "clean_read"}
    assert "NOT corpus WER" in known["summary"]["labels"]["unweighted_mean_of_set_wers"]


def test_mixed_reference_tracks_refused():
    rows = [row("a", "one two", "one two"),
            row("b", "three four", "three four", reference_track="cleaned")]
    with pytest.raises(S.MixedReferenceTracks):
        score(manifest(rows))


def test_pool_across_tracks_refused():
    a = score(manifest([row("a", "one two", "one two")]))["records"][0]
    b = score(manifest([row("a", "one two", "one two", reference_track="cleaned")]))["records"][0]
    with pytest.raises(S.MixedReferenceTracks):
        S.pool_records([a, b])


# --- mutation cases ------------------------------------------------------------

def test_bracket_wrapped_error_is_not_a_match():
    r = score(manifest([row("u", "right answer", "[wrong guess]")]))
    u = per_utt(r, "u")
    assert u["class"] == "unknown_marker"
    assert "[wrong guess]" in u["unknown_markers"]
    assert u["errors"] == 2  # aligned honestly; excluded, never counted as a match
    p = r["records"][0]
    assert p["n"] == 0 and p["unknown_marker_count"] == 1 and p["wer"] is None


def test_closed_marker_hypothesis_is_empty_not_match():
    u = per_utt(score(manifest([row("u", "music", "[MUSIC]")])), "u")
    assert u["class"] == "empty"


def test_reference_aware_rewrite_flag_refused():
    with pytest.raises(S.ReferenceAwareRewriteRefused):
        score(manifest([row("u", "a b", "a b", rewritten_with_reference=True)]))


def test_rewrite_hook_in_row_or_arm_refused():
    with pytest.raises(S.ReferenceAwareRewriteRefused):
        score(manifest([row("u", "a b", "a b", postprocess=lambda h, r: r)]))
    arm = {"arm_id": "x", "arm_kind": S.ARM_KIND_OFFLINE, "runner_id": "r",
           "decode_options": lambda r: r}
    with pytest.raises(S.ReferenceAwareRewriteRefused):
        score(manifest([row("u", "a b", "a b")], arm=arm))


def test_scorer_entry_points_take_no_hook():
    for fn in (S.score_arm, S.score_utterance):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"rewrite", "hook", "postprocess", "hypothesis_rewrite",
                             "reference", "normaliser"}


def test_uncovered_numeral_is_could_not_check_and_excluded():
    r = score(manifest([row("u", "in nineteen oh five", "in 1905"),
                        row("v", "in nineteen twenty", "in 1920")]))
    assert per_utt(r, "u")["class"] == "numeral_uncovered"
    assert per_utt(r, "u")["numeral_uncovered_tokens"] == ["1905"]
    assert per_utt(r, "v")["class"] == "ok" and per_utt(r, "v")["errors"] == 0
    p = r["records"][0]
    assert p["numeral_uncovered_count"] == 1 and p["n"] == 1 and p["wer"] == 0.0


# --- (d) chunk join and VOID -----------------------------------------------------

def test_chunks_joined_in_index_order():
    rw = row("u", "a b c d", hypothesis_chunks=["c d", "a b"], chunk_indices=[1, 0])
    u = per_utt(score(manifest([rw], segmentation_mode="vad_30s")), "u")
    assert u["class"] == "ok" and u["errors"] == 0


@pytest.mark.parametrize("chunks,indices,reason", [
    (["a", "c"], [0, 2], "non_contiguous_chunk_indices"),
    (["a", "b"], [0, 0], "non_contiguous_chunk_indices"),
    (["a", "b"], [0], "chunk_count_mismatch"),
    (["a", "b"], [1, 2], "non_contiguous_chunk_indices"),
])
def test_bad_chunk_indices_void_as_harness_fault(chunks, indices, reason):
    rows = [row("bad", "a b c", hypothesis_chunks=chunks, chunk_indices=indices),
            row("good", "x y", "x y")]
    r = score(manifest(rows, segmentation_mode="vad_30s"))
    u = per_utt(r, "bad")
    assert (u["class"], u["void_reason"]) == ("VOID", reason)
    p = r["records"][0]
    assert p["harness_fault_void_count"] == 1
    assert p["n"] == 1 and p["errors"] == 0  # VOID never scored as an error
    assert sum(p["taxonomy"].values()) == 1
    assert p["taxonomy_check"]["outcome"] == "PASS"


def test_long_clip_unsegmented_is_void():
    rows = [row("u", "a b", "a b", duration_s=45.0), row("v", "c", "c")]
    r = score(manifest(rows, segmentation_mode="vad_30s"))
    assert per_utt(r, "u")["void_reason"] == "long_clip_not_segmented"
    assert r["records"][0]["segmentation_mode"] == "vad_30s"


# --- taxonomy ---------------------------------------------------------------------

def test_decode_error_and_empty_are_correctness_fail():
    rows = [row("a", "one two", "one two"), row("b", "three", "", ),
            row("c", "four", "four", decode_status="error")]
    r = score(manifest(rows))
    p = r["records"][0]
    assert p["taxonomy"]["empty"] == 1 and p["taxonomy"]["decode_error"] == 1
    assert p["correctness_verdict"] == "FAIL"
    assert p["taxonomy_check"]["outcome"] == "FAIL"
    assert p["wer"] == 0.0 and p["n"] == 1  # failures are not averaged into the rate


def test_repetition_envelope_derived_never_defaulted():
    rows = [row("a", "one two", "one two one two one two"), row("b", "x y", "x y")]
    r0 = score(manifest(rows))
    assert r0["records"][0]["repetition_check"] == "COULD_NOT_CHECK"
    assert per_utt(r0, "a")["class"] == "ok"
    r1 = score(manifest(rows), anchor_hyp_to_ref_ratios=[1.0, 1.25])
    assert r1["records"][0]["repetition_envelope"] == 1.25
    assert per_utt(r1, "a")["class"] == "repetition_loop"
    assert S.hyp_to_ref_ratios(r1["per_utterance"]) == [1.0]


def test_normaliser_contract_failure_voids_window(monkeypatch):
    monkeypatch.setattr(stt_norm, "contract_check", lambda nid: whisper_stt.schemas.Check(
        whisper_stt.schemas.FAIL, ("tampered",)))
    with pytest.raises(S.WindowVoid):
        score(manifest([row("a", "x", "x")]))


# --- (b) v2 candidate column, DP-B, German compounds ----------------------------------

def test_v2_records_are_candidate_and_refused_for_pooling_with_v1():
    m = manifest([row("a", "one two", "one two")])
    v1 = score(m, "v1")["records"][0]
    v2 = score(m, "v2_candidate")["records"][0]
    assert v2["candidate"] is True and v2["comparable_with"] == []
    assert v2["manifest_sha256"] == v1["manifest_sha256"]  # rescoring, no re-inference
    assert v2["normaliser_id"] == "stt_norm/v2" != v1["normaliser_id"]
    for fn in (lambda: S.pool_records([v1, v2]), lambda: S.difference_records(v1, v2),
               lambda: S.check_comparable(v1, v2)):
        with pytest.raises(S.IncomparableRecords):
            fn()


def test_secondary_column_refused_against_v2_primary():
    m = manifest([row("a", "euh oui", "oui", language="fr")])
    v2 = score(m, "v2_candidate")["records"][0]
    sec = score(m, "secondary")["records"][0]
    assert sec["secondary"] is True and sec["errors"] == 0
    assert v2["errors"] == 1  # DP-B: the filler is an ordinary word in v2-primary
    with pytest.raises(S.IncomparableRecords):
        S.pool_records([v2, sec])


def test_fillers_in_ref_flagged_not_removed():
    m = manifest([row("a", "um so uh yes", "so yes"), row("b", "plain", "plain")])
    for col in ("v1", "v2_candidate"):
        p = score(m, col)["records"][0]
        assert p["fillers_in_ref"] == 2 and p["fillers_in_ref_utterances"] == 1
        assert p["errors"] == 2


def test_v2_elision_typographic_apostrophe():
    m = manifest([row("a", "l'homme", "l’homme", language="fr")])
    assert score(m, "v1")["records"][0]["errors"] == 2
    assert score(m, "v2_candidate")["records"][0]["errors"] == 0


def test_german_compound_class_counted_not_forgiven():
    m = manifest([row("a", "der Bahnhof ist hier", "der Bahn Hof ist hier", language="de"),
                  row("b", "das Haus Tür", "das Haustür", language="de")])
    v2 = score(m, "v2_candidate")
    assert per_utt(v2, "a")["de_compound_split_join"] == 1
    assert per_utt(v2, "b")["de_compound_split_join"] == 1
    p = v2["records"][0]
    assert p["de_compound_split_join"] == 2 and p["errors"] == 4
    assert score(m, "v1")["records"][0]["de_compound_split_join"] is None


# --- (c) record shape -----------------------------------------------------------

def test_records_carry_every_vb_speech_conv_1b_field(known):
    for rec in known["records"]:
        assert S.check_record_fields(rec) == []
        assert rec["metric_direction"] == "lower" and rec["metric"] == "stt_corpus_wer"
        assert rec["normaliser_sha256"] == stt_norm.normaliser_sha256("stt_norm/v1")
        assert rec["scorer_sha256"] == S.scorer_sha256()
        assert rec["language_mode"] == "forced" and rec["language"] == "en"


def test_manifest_and_corpus_hashes():
    rows = known_answer_rows()
    r = score(manifest(rows))["records"][0]
    canon = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert r["manifest_sha256"] == hashlib.sha256(canon.encode()).hexdigest()
    assert r["corpus_key"] == "synth@" + r["corpus_manifest_sha256"][:12]
    other_hyp = copy.deepcopy(rows)
    other_hyp[0]["hypothesis_raw"] = "different"
    r2 = score(manifest(other_hyp))["records"][0]
    assert r2["manifest_sha256"] != r["manifest_sha256"]
    assert r2["corpus_key"] == r["corpus_key"]  # same corpus across arms


def test_records_carry_no_per_utterance_values(known):
    blob = json.dumps(known["records"])
    assert "u000" not in blob and known["per_utterance"][0]["pcm_sha256"] not in blob
    for rec in known["records"]:
        assert not set(rec) & {"utterance_id", "pcm_sha256", "reference_normalised",
                               "hypothesis_normalised", "per_utterance"}


def test_bad_language_mode_refused():
    with pytest.raises(S.ScorerError):
        score(manifest([row("a", "x", "x")], language_mode="guess"))


# --- (f) unk ----------------------------------------------------------------------

def parakeet(arm_id, unk_ban, **kw):
    a = {"arm_id": arm_id, "arm_kind": S.ARM_KIND_OFFLINE, "runner_id": "nemo",
         "model_family": "parakeet-tdt"}
    if unk_ban is not None:
        a["unk_ban"] = unk_ban
    a.update(kw)
    return a


def test_parakeet_arm_must_declare_unk_ban():
    with pytest.raises(S.ScorerError):
        score(manifest([row("a", "x", "x")], arm=parakeet("p", None)))


def test_unk_count_and_ban_violation_recorded():
    r = score(manifest([row("a", "the cat", "the <unk>")], arm=parakeet("p", True)))
    p = r["records"][0]
    assert p["unk_count"] == 1 and p["unk_ban"] is True and p["unk_ban_violated"] is True
    assert p["errors"] == 1  # <unk> scores as a wrong token, never excluded


def test_unk_ban_symmetry_across_parakeet_arms():
    S.check_unk_ban_symmetry([parakeet("a", True), parakeet("b", True),
                              {"arm_id": "w", "model_family": "whisper"}])
    with pytest.raises(S.ScorerError):
        S.check_unk_ban_symmetry([parakeet("a", True), parakeet("b", False)])


# --- (g) arm kind / delay ---------------------------------------------------------

def streaming(arm_id):
    return {"arm_id": arm_id, "arm_kind": S.ARM_KIND_STREAMING, "runner_id": "nemo",
            "chunk_label": "160ms", "algorithmic_delay_ms": 160, "att_context_right": 1}


def test_streaming_arm_requires_delay_fields():
    arm = streaming("s")
    del arm["algorithmic_delay_ms"]
    with pytest.raises(S.ScorerError):
        score(manifest([row("a", "x", "x")], arm=arm))


def test_offline_arm_refused_in_streaming_column():
    off = score(manifest([row("a", "x", "x")]))["records"][0]
    st = score(manifest([row("a", "x", "x")], arm=streaming("s")))["records"][0]
    rows = S.delay_table([st], arm_kind=S.ARM_KIND_STREAMING)
    assert rows[0]["algorithmic_delay_ms"] == 160 and rows[0]["chunk_label"] == "160ms"
    with pytest.raises(S.ArmKindMismatch):
        S.delay_table([st, off], arm_kind=S.ARM_KIND_STREAMING)
    with pytest.raises(S.ArmKindMismatch):
        S.delay_table([st], arm_kind=S.ARM_KIND_OFFLINE)


# --- (h) rank tables --------------------------------------------------------------

def arm_result(arm_id, hyps, set_id="pub"):
    rows = [row(f"u{i}", "a b c d", h, set_id=set_id) for i, h in enumerate(hyps)]
    return score(manifest(rows, arm={"arm_id": arm_id, "arm_kind": S.ARM_KIND_OFFLINE,
                                     "runner_id": "r"}))["records"]


def test_rank_table_ties_and_fail_unranked():
    recs = (arm_result("A", ["a b c d", "a b c x"]) + arm_result("B", ["a b c d", "a b x d"])
            + arm_result("C", ["a b c d", "a b c d"]) + arm_result("D", ["a b c d", ""]))
    t = S.rank_table(recs, set_id="pub")
    assert [(r["rank"], r["arm_id"]) for r in t["rows"]] == [(1, "C"), (2, "A"), (2, "B")]
    assert t["unranked"] == ["D"]  # correctness FAIL gets no rank, not a penalised one


def test_rank_delta_public_vs_private():
    pub = S.rank_table(arm_result("A", ["a b c d"]) + arm_result("B", ["a b c x"]),
                       set_id="pub")
    prv = S.rank_table(arm_result("A", ["a x x d"], "prv") + arm_result("B", ["a b c d"], "prv"),
                       set_id="prv")
    d = S.rank_delta_public_vs_private(pub, prv)
    assert d["deltas"] == {"A": 1, "B": -1} and d["not_in_both"] == []
