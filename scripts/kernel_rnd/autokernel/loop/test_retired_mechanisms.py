import json
from types import SimpleNamespace

from autokernel.loop import dispatch_guard as D
from autokernel.loop import retired_mechanisms as R

SURFACE = "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp"
SYMBOL = "mul_mat_qX_K_q8_K_IQ_1"


def proposal(mechanism, symbol=SYMBOL, surface=SURFACE):
    return SimpleNamespace(mechanism_id=mechanism, target_symbol=symbol, target_surface=surface)


def zmm_entry(**overrides):
    entry = {"target_surface": SURFACE, "target_symbol": SYMBOL,
             "mechanism": "akm-cpu-iq3s-iq1-zmm-dot-gather", "status": "in-flight",
             "receipt": "fc71ff076", "reopen_if": "the branch is abandoned",
             "enforcement": "hard", "mechanism_tokens": ["zmm"]}
    return {**entry, **overrides}


def write(tmp_path, *entries):
    path = tmp_path / R.LIST_NAME
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return path


def guard(hypothesis, path):
    return D.characterised_reason(hypothesis, {
        "epoch_sha256": "e1", "prior_experiments": [], "retired_mechanisms_path": path})


def test_zmm_pass_refused_by_fc71ff076_entry(tmp_path):
    reason = guard(proposal("akm-cpu-iq3s-iq1-zmm-pass"), write(tmp_path, zmm_entry()))
    assert "retired_mechanism" in reason and "fc71ff076" in reason and "reopen_if" in reason


def test_different_mechanism_on_same_symbol_allowed(tmp_path):
    path = write(tmp_path, zmm_entry())
    assert guard(proposal("akm-cpu-iq3s-iq1-sign-table-prefetch"), path) is None


def test_two_mechanisms_on_one_symbol_each_match_only_their_own(tmp_path):
    path = write(tmp_path, zmm_entry(), zmm_entry(mechanism="akm-x-wide-route",
                                                  mechanism_tokens=["wide", "route"]))
    assert "zmm-dot" in guard(proposal("akm-zmm-pass"), path)
    assert "akm-x-wide-route" in guard(proposal("akm-wide-route-v2"), path)
    assert guard(proposal("akm-wide-only"), path) is None


def test_other_symbol_or_surface_allowed(tmp_path):
    path = write(tmp_path, zmm_entry())
    assert guard(proposal("akm-zmm", symbol="other_fn"), path) is None
    assert guard(proposal("akm-zmm", surface="ggml/src/ggml-cpu/other.cpp"), path) is None


def test_advisory_entry_not_refused(tmp_path):
    path = write(tmp_path, zmm_entry(enforcement="advisory", status="tried-null"))
    assert guard(proposal("akm-cpu-iq3s-iq1-zmm-pass"), path) is None


def test_missing_list_fails_open_and_logs(tmp_path):
    notes = []
    assert R.refusal(proposal("akm-zmm"), tmp_path / "absent.jsonl", note=notes.append) is None
    assert "guard=OPEN" in notes[0]


def test_corrupt_list_fails_open_and_logs(tmp_path):
    path = tmp_path / R.LIST_NAME
    path.write_text("{not json\n", encoding="utf-8")
    notes = []
    assert R.refusal(proposal("akm-zmm"), path, note=notes.append) is None
    assert "guard=OPEN" in notes[0]


def test_no_path_in_context_is_silent(capsys):
    assert D.characterised_reason(proposal("akm-zmm"),
                                  {"epoch_sha256": "e1", "prior_experiments": []}) is None
    assert capsys.readouterr().out == ""


def test_negatives_inconclusive_stay_advisory_with_reopen():
    rows = {e["mechanism"]: e for e in R._from_negatives(R.SEEDS)}
    iq4 = rows["akm-c9-q38fn-iq4nl-moe-iqk"]
    assert iq4["enforcement"] == "advisory" and iq4["reopen_if"]
    assert rows["akm-c9-gv-fattn-more-splits"]["enforcement"] == "advisory"
    assert rows["akm-hist-q8-prefetch"]["enforcement"] == "advisory"
    assert all(e["reopen_if"] for e in rows.values())


def test_shipped_manifest_shape():
    body = json.loads((R.SEEDS / "retired_manifest.json").read_text(encoding="utf-8"))
    by_id = {e["mechanism"]: e for e in body["entries"]}
    assert by_id["akm-cpu-iq3s-iq1-zmm-dot-gather"]["commit"] == "fc71ff076"
    assert by_id["akm-cpu-iq4nl-iqk-route-prefill"]["enforcement"] == "advisory"
    assert by_id["akm-cpu-iq3s-avx512vl-sign-retest"]["enforcement"] == "advisory"
    assert all(e["status"] in R.STATUSES and e["reopen_if"] for e in body["entries"])


def test_render_separates_hard_and_advisory():
    text = R.render([zmm_entry(), zmm_entry(mechanism="soft-one", enforcement="advisory")], "c9")
    assert text.index("zmm-dot-gather") < text.index("## Advisory") < text.index("soft-one")
