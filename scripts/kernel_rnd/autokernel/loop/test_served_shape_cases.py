"""Unit tests for the 2026-10-06 served-shape test-backend-ops case set
(`served_shape_cases.py`) and its gates.py wiring (`check_served_shape_case_set`,
`ppl_contract_op_nmse`'s `served_shape_manifest` addendum).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from . import gates
from . import served_shape_cases as ssc

ANCHOR_NMSE = {(shape.name, t, n): 1e-5 for shape in ssc.SERVED_SHAPES
               for t in ssc.WITNESS_TYPES for n in ssc.SERVED_WIDTHS}


def _recipe():
    return object()


# ---- shapes / provenance ----------------------------------------------------------

def test_served_shapes_match_the_2026_10_06_gguf_header_read():
    """Real dims read off the lanes' own GGUF headers (module docstring PROVENANCE),
    never invented. DS41 (deepseek41) expert: embedding_length=5120,
    expert_feed_forward_length=2304, expert_count=128, expert_used_count=3. Q38FN
    (qwen35) dense: embedding_length=5120, feed_forward_length=17408."""
    by_name = {s.name: s for s in ssc.SERVED_SHAPES}
    assert by_name["ds41_expert_gate_up"] == ssc.ServedShape(
        "ds41_expert_gate_up", "MUL_MAT_ID", k=5120, m=2304, n_mats=128, n_used=3)
    assert by_name["ds41_expert_down"] == ssc.ServedShape(
        "ds41_expert_down", "MUL_MAT_ID", k=2304, m=5120, n_mats=128, n_used=3)
    assert by_name["ds41_dense_shexp_gate_up"].op == "MUL_MAT"
    assert by_name["q38fn_dense_ffn_gate_up"] == ssc.ServedShape(
        "q38fn_dense_ffn_gate_up", "MUL_MAT", k=5120, m=17408)
    assert by_name["q38fn_dense_ffn_down"] == ssc.ServedShape(
        "q38fn_dense_ffn_down", "MUL_MAT", k=17408, m=5120)
    assert {s.op for s in ssc.SERVED_SHAPES} == {"MUL_MAT", "MUL_MAT_ID"}


def test_served_widths_match_the_ppl_contract_served_widths():
    """Duplicated, not imported (import-cycle-free of gates), but must never drift."""
    assert ssc.SERVED_WIDTHS == gates.PPL_CONTRACT_SERVED_WIDTHS


def test_shape_constructor_refuses_malformed_moe_fields():
    with pytest.raises(ValueError, match="n_used must be"):
        ssc.ServedShape("bad", "MUL_MAT_ID", k=1, m=1, n_mats=4, n_used=5)
    with pytest.raises(ValueError, match="n_mats/n_used"):
        ssc.ServedShape("bad", "MUL_MAT", k=1, m=1, n_mats=4, n_used=1)
    with pytest.raises(ValueError, match="op must be"):
        ssc.ServedShape("bad", "SOFT_MAX", k=1, m=1)


# ---- threshold formula --------------------------------------------------------------

def test_tightened_bound_scales_caps_and_floors():
    assert ssc.tightened_nmse_bound(1e-6, factor=3.0) == pytest.approx(3e-6)
    # Capped well below the generic 5e-4 bound, never above it...
    assert ssc.tightened_nmse_bound(5e-5, factor=3.0) == ssc.SERVED_SHAPE_NMSE_CAP
    # ...but an anchor already at/above the cap is REFUSED, never clipped below its
    # own error (re-review 2026-10-06).
    with pytest.raises(ValueError, match="do not clip"):
        ssc.tightened_nmse_bound(1.0, factor=3.0)
    with pytest.raises(ValueError):
        ssc.tightened_nmse_bound(ssc.SERVED_SHAPE_NMSE_CAP)
    assert ssc.SERVED_SHAPE_NMSE_CAP < 5e-4
    # A measured-zero anchor NMSE must not produce a zero bound.
    assert ssc.tightened_nmse_bound(0.0) == ssc.SERVED_SHAPE_NMSE_FLOOR


def test_tightened_bound_refuses_bad_inputs():
    with pytest.raises(ValueError):
        ssc.tightened_nmse_bound(-1.0)
    with pytest.raises(ValueError):
        ssc.tightened_nmse_bound(float("nan"))
    with pytest.raises(ValueError):
        ssc.tightened_nmse_bound(1e-5, factor=0.0)


# ---- case set / vars / regex / patch block -----------------------------------------

def test_case_set_is_the_full_cross_product():
    cases = ssc.case_set(ANCHOR_NMSE)
    assert len(cases) == len(ssc.SERVED_SHAPES) * len(ssc.WITNESS_TYPES) * len(ssc.SERVED_WIDTHS)
    assert {c.n for c in cases} == set(ssc.SERVED_WIDTHS)
    assert {c.type_a for c in cases} == set(ssc.WITNESS_TYPES)
    assert all(0 < c.max_nmse <= ssc.SERVED_SHAPE_NMSE_CAP for c in cases)


def test_case_set_refuses_a_missing_anchor_measurement():
    partial = dict(ANCHOR_NMSE)
    del partial[("ds41_expert_gate_up", "IQ3_S", 3)]   # one WIDTH missing refuses
    with pytest.raises(KeyError, match="ds41_expert_gate_up"):
        ssc.case_set(partial)


def test_vars_and_cpp_agree_with_the_real_test_backend_ops_constructors():
    """test-backend-ops prints `ggml_type_name()` spellings (`q4_K`, `iq3_s`, `f32`),
    and the generated subclasses append `max_nmse` to the base `vars()` (review
    2026-10-06: the first cut spelled `IQ3_S`/`F32` and the base vars() had no
    max_nmse, so the anchored selector matched 0 real cases)."""
    cases = ssc.case_set(ANCHOR_NMSE)
    dense = next(c for c in cases if c.shape.op == "MUL_MAT")
    expert = next(c for c in cases if c.shape.op == "MUL_MAT_ID")
    name = ssc.GGML_TYPE_NAMES
    assert set(name) == set(ssc.WITNESS_TYPES)
    assert dense.vars() == (f"type_a={name[dense.type_a]},type_b=f32,m={dense.shape.m},"
                            f"n={dense.n},k={dense.shape.k},bs=[1,1],nr=[1,1],"
                            f"per=[0,1,2,3],k_v=0,o=1,max_nmse={dense.max_nmse:g}")
    assert "test_mul_mat_served_shape(" in dense.cpp()
    assert f"GGML_TYPE_{dense.type_a}" in dense.cpp()
    assert expert.vars() == (
        f"type_a={name[expert.type_a]},type_b=f32,n_mats={expert.shape.n_mats},"
        f"n_used={expert.shape.n_used},b=0,m={expert.shape.m},n={expert.n},"
        f"k={expert.shape.k},max_nmse={expert.max_nmse:g}")
    assert "test_mul_mat_id_served_shape(" in expert.cpp()
    block = ssc.backend_ops_patch_block(cases[:1])
    assert block.count('",max_nmse=" + autokernel_served_shape_nmse_str(max_nmse)') == 2


def test_type_names_match_the_real_ggml_type_table():
    tree = Path("/mnt/raid0/llm/llama.cpp-experimental-cor-b0ba1d427-20261005")
    source = tree / "ggml/src/ggml.c"
    if not source.is_file():
        pytest.skip("champion tree not present")
    text = source.read_text(encoding="utf-8", errors="replace")
    for spelled in ssc.GGML_TYPE_NAMES.values():
        assert f'.type_name                = "{spelled}"' in text, spelled
    vars_src = (tree / "tests/test-backend-ops.cpp").read_text(encoding="utf-8")
    assert "return VARS_TO_STR10(type_a, type_b, m, n, k, bs, nr, per, k_v, o);" in vars_src
    assert "return VARS_TO_STR8(type_a, type_b, n_mats, n_used, b, m, n, k);" in vars_src


def test_case_set_regex_selects_exactly_its_cases():
    cases = ssc.case_set(ANCHOR_NMSE)[:5]
    regex = re.compile(ssc.case_set_regex(cases))
    assert all(regex.search(c.vars()) for c in cases)
    other = ssc.case_set(ANCHOR_NMSE)[5]
    assert not regex.search(other.vars())
    assert not regex.search("x" + cases[0].vars())


def test_backend_ops_patch_block_carries_every_case_and_both_subclasses():
    cases = ssc.case_set(ANCHOR_NMSE)[:3]
    block = ssc.backend_ops_patch_block(cases)
    assert "struct test_mul_mat_served_shape : public test_mul_mat" in block
    assert "struct test_mul_mat_id_served_shape : public test_mul_mat_id" in block
    assert f'"{ssc.CASE_SET_ID}"' in block and f'"{ssc.CASE_SET_ENV}"' in block
    assert all(c.cpp() in block for c in cases)


def test_binary_has_case_set_scans_for_the_literal(tmp_path):
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    binary = build / "bin" / "test-backend-ops"
    binary.write_bytes(b"\0ELF\0")
    assert ssc.binary_has_case_set(build) is False
    binary.write_bytes(b"\0ELF" + ssc.CASE_SET_ID.encode() + b"\0")
    assert ssc.binary_has_case_set(build) is True
    assert ssc.binary_has_case_set(tmp_path / "missing") is False


# ---- manifest -----------------------------------------------------------------------

def test_manifest_round_trips(tmp_path):
    cases = ssc.case_set(ANCHOR_NMSE)
    path = tmp_path / "served_shape" / "manifest.json"
    ssc.write_manifest(path, cases)
    assert ssc.load_manifest(path) == cases


def test_manifest_refuses_missing_file(tmp_path):
    with pytest.raises(ssc.ManifestRefused, match="cannot read"):
        ssc.load_manifest(tmp_path / "absent.json")


def test_manifest_refuses_malformed_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ssc.ManifestRefused, match="not valid JSON"):
        ssc.load_manifest(path)


def test_manifest_refuses_wrong_schema(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema": "wrong", "case_set_id": "x", "cases": []}', encoding="utf-8")
    with pytest.raises(ssc.ManifestRefused, match="wrong shape or schema"):
        ssc.load_manifest(path)


def test_manifest_refuses_unknown_shape_name(tmp_path):
    cases = ssc.case_set(ANCHOR_NMSE)
    path = tmp_path / "manifest.json"
    ssc.write_manifest(path, cases)
    import json
    body = json.loads(path.read_text())
    body["cases"][0]["shape_name"] = "nonexistent_shape"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ssc.ManifestRefused, match="unknown shape"):
        ssc.load_manifest(path)


def test_manifest_refuses_an_out_of_bound_max_nmse(tmp_path):
    cases = ssc.case_set(ANCHOR_NMSE)
    path = tmp_path / "manifest.json"
    ssc.write_manifest(path, cases)
    import json
    body = json.loads(path.read_text())
    body["cases"][0]["max_nmse"] = 1.0   # above SERVED_SHAPE_NMSE_CAP
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ssc.ManifestRefused, match="out of bounds"):
        ssc.load_manifest(path)


def test_manifest_refuses_an_incomplete_case_set(tmp_path):
    cases = ssc.case_set(ANCHOR_NMSE)[:-1]   # drop one case
    path = tmp_path / "manifest.json"
    ssc.write_manifest(path, cases)
    with pytest.raises(ssc.ManifestRefused, match="not the full"):
        ssc.load_manifest(path)


# ---- gates.py wiring ------------------------------------------------------------------

def test_check_served_shape_case_set_fails_closed_on_missing_manifest(tmp_path):
    verdict = gates.check_served_shape_case_set(
        tmp_path / "build", resolved_recipe=_recipe(), manifest_path=tmp_path / "absent.json")
    assert verdict.passed is False
    assert verdict.gate == "served_shape_case_set"
    assert "not available" in verdict.reason


def test_check_served_shape_case_set_fails_closed_on_missing_binary_literal(tmp_path):
    cases = ssc.case_set(ANCHOR_NMSE)
    manifest = tmp_path / "manifest.json"
    ssc.write_manifest(manifest, cases)
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "test-backend-ops").write_bytes(b"\0ELF\0")
    verdict = gates.check_served_shape_case_set(
        build, resolved_recipe=_recipe(), manifest_path=manifest)
    assert verdict.passed is False
    assert "does not carry" in verdict.reason


def test_check_served_shape_case_set_runs_op_correctness_when_available(tmp_path, monkeypatch):
    cases = ssc.case_set(ANCHOR_NMSE)
    manifest = tmp_path / "manifest.json"
    ssc.write_manifest(manifest, cases)
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "test-backend-ops").write_bytes(
        b"\0ELF" + ssc.CASE_SET_ID.encode() + b"\0")
    seen = {}

    def fake_op_correctness(build_dir, *, op, backend, resolved_recipe, params_filter,
                            environment_overrides, expected_cases):
        seen.update(op=op, params_filter=params_filter,
                   environment_overrides=environment_overrides, expected_cases=expected_cases)
        return gates.Verdict("correctness", True, "ok")

    monkeypatch.setattr(gates, "op_correctness", fake_op_correctness)
    verdict = gates.check_served_shape_case_set(
        build, resolved_recipe=_recipe(), manifest_path=manifest)
    assert verdict.passed is True
    assert seen["expected_cases"] == len(cases)
    assert seen["environment_overrides"] == ((ssc.CASE_SET_ENV, ssc.CASE_SET_ID),)
    assert "MUL_MAT" in seen["op"] and "MUL_MAT_ID" in seen["op"]


def test_ppl_contract_op_nmse_skips_served_shape_when_manifest_is_none():
    """Backward compatibility: every caller before the 2026-10-06 review (and every
    unit test not explicitly exercising this addendum) passes no manifest, so layer
    (a) behaves exactly as before."""
    def op_correctness(build, *, op, backend, resolved_recipe, params_filter):
        return gates.Verdict("correctness", True, "ok")

    verdict = gates.ppl_contract_op_nmse(
        "build", resolved_recipe=_recipe(), _op_correctness=op_correctness)
    assert verdict.passed is True
    assert "served-shape" not in verdict.reason


def test_ppl_contract_op_nmse_fails_closed_when_served_shape_check_fails():
    def op_correctness(build, *, op, backend, resolved_recipe, params_filter):
        return gates.Verdict("correctness", True, "ok")

    def failing_served_shape_check(build, *, resolved_recipe, manifest_path):
        return gates.Verdict("served_shape_case_set", False, "manifest refused: boom")

    verdict = gates.ppl_contract_op_nmse(
        "build", resolved_recipe=_recipe(), served_shape_manifest="manifest.json",
        _op_correctness=op_correctness,
        _check_served_shape_case_set=failing_served_shape_check)
    assert verdict.passed is False
    assert "served-shape suite refused" in verdict.reason
    assert "boom" in verdict.reason


def test_ppl_contract_op_nmse_passes_when_both_the_generic_and_served_shape_suites_pass():
    def op_correctness(build, *, op, backend, resolved_recipe, params_filter):
        return gates.Verdict("correctness", True, "ok")

    def passing_served_shape_check(build, *, resolved_recipe, manifest_path):
        return gates.Verdict("served_shape_case_set", True, "ok")

    verdict = gates.ppl_contract_op_nmse(
        "build", resolved_recipe=_recipe(), served_shape_manifest="manifest.json",
        _op_correctness=op_correctness,
        _check_served_shape_case_set=passing_served_shape_check)
    assert verdict.passed is True
    assert "served-shape suite passed" in verdict.reason


def test_direct_launch_anchor_build_keeps_widened_targets():
    """Review 2026-10-06: run.py's build_champion forced PROMOTION_TARGETS on a direct
    launch, silently DROPPING the ppl_contract tools `promote_anchor` asked for (while
    provenance.json recorded them). It must take the union instead."""
    source = (Path(__file__).parent / "run.py").read_text(encoding="utf-8")
    body = source[source.index("    def build_champion("):source.index("    def build_baseline(")]
    assert "targets=gates.PROMOTION_TARGETS if direct_launch else targets" not in body
    assert "dict.fromkeys((*gates.PROMOTION_TARGETS," in body
    fold = source[source.index("    def ppl_contract_fold_check("):]
    fold = fold[:fold.index("    def gpu_reading(")]
    assert "served_shape_manifest=" in fold


def test_manifest_refuses_duplicates_and_partial_corpora(tmp_path):
    """Re-review 2026-10-06: a row count accepted N copies of one case."""
    cases = ssc.case_set(ANCHOR_NMSE)
    path = tmp_path / "m.json"
    ssc.write_manifest(path, cases)
    assert len(ssc.load_manifest(path)) == len(cases)
    ssc.write_manifest(path, (cases[0],) * len(cases))
    with pytest.raises(ssc.ManifestRefused):
        ssc.load_manifest(path)
    ssc.write_manifest(path, cases[:-1] + (cases[0],))
    with pytest.raises(ssc.ManifestRefused, match="repeats"):
        ssc.load_manifest(path)


def test_witness_types_cover_every_type_the_iqk_whitelist_admits():
    tree = Path("/mnt/raid0/llm/llama.cpp-experimental-cor-b0ba1d427-20261005")
    src = tree / "ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp"
    if not src.is_file():
        pytest.skip("champion tree not present")
    text = src.read_text(encoding="utf-8")
    body = text[text.index("constexpr bool iqk_typeA_supported(int t) {"):]
    body = body[:body.index("default:")]
    admitted = set(re.findall(r"case GGML_TYPE_(\w+):", body))
    assert admitted and admitted <= set(ssc.WITNESS_TYPES), admitted - set(ssc.WITNESS_TYPES)
