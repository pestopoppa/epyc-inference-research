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


def _served_recipe():
    from types import SimpleNamespace
    return SimpleNamespace(topology_prefix=("taskset", "-c", "0-95"),
                           command_argv=("llama-server", "-t", "48", "-tb", "48"),
                           launch_env=(), backend="cpu")


# ---- shapes / provenance ----------------------------------------------------------

def test_served_shapes_derive_from_each_lanes_served_gguf():
    """Round-13: dims and expert counts come from the SERVED models' GGUF headers
    (DS41 DeepSeek-V4.1-Flash-Q4: 384 experts / 6 used, 5120 x 2304; Q38FN
    Qwen3.8-Flash-Next: 512 / 10, 2560 x 640) -- never a drafter or a sibling model."""
    by_name = {s.name: s for s in ssc.SERVED_SHAPES}
    assert by_name["ds41_expert_gate_up"] == ssc.ServedShape(
        "ds41_expert_gate_up", "MUL_MAT_ID", k=5120, m=2304, n_mats=8, n_used=6)
    assert by_name["ds41_expert_down"] == ssc.ServedShape(
        "ds41_expert_down", "MUL_MAT_ID", k=2304, m=5120, n_mats=8, n_used=6)
    assert by_name["q38fn_expert_gate_up"] == ssc.ServedShape(
        "q38fn_expert_gate_up", "MUL_MAT_ID", k=2560, m=640, n_mats=16, n_used=10)
    assert by_name["q38fn_expert_down"] == ssc.ServedShape(
        "q38fn_expert_down", "MUL_MAT_ID", k=640, m=2560, n_mats=16, n_used=10)
    assert by_name["q38fn_shexp_gate_up"].op == "MUL_MAT"
    assert {s.op for s in ssc.SERVED_SHAPES} == {"MUL_MAT", "MUL_MAT_ID"}
    assert all(s.n_used <= s.n_mats for s in ssc.SERVED_SHAPES)


@pytest.mark.parametrize("lane", sorted(ssc.LANE_PROFILES))
def test_lane_profiles_match_a_fresh_gguf_header_read(lane):
    profile = ssc.LANE_PROFILES[lane]
    model = Path(profile.model)
    if not model.is_file() or not Path("/mnt/raid0/llm/llama.cpp/gguf-py").is_dir():
        pytest.skip("served GGUF or gguf-py not present")
    shards = sorted(model.parent.glob(model.name.replace("00001-of", "*-of"))) or [model]
    derived = ssc.moe_profile_from_gguf(shards, lane)
    for field in ("expert_count", "expert_used", "hidden", "expert_ff", "shexp_ff"):
        assert getattr(derived, field) == getattr(profile, field), field
    assert set(derived.expert_types) == set(profile.expert_types)


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
    # Operator 2026-10-06: the cap is the generic test-backend-ops bound (5e-4); tight
    # wherever the anchor is accurate, never looser than the generic check...
    assert ssc.SERVED_SHAPE_NMSE_CAP == 5e-4
    assert ssc.tightened_nmse_bound(1e-4, factor=3.0) == pytest.approx(3e-4)
    assert ssc.tightened_nmse_bound(2e-4, factor=3.0) == ssc.SERVED_SHAPE_NMSE_CAP
    # ...an anchor already at/above it is REFUSED by default, never clipped below its
    # own error...
    with pytest.raises(ValueError, match="do not clip"):
        ssc.tightened_nmse_bound(1.0, factor=3.0)
    with pytest.raises(ValueError):
        ssc.tightened_nmse_bound(ssc.SERVED_SHAPE_NMSE_CAP)
    # ...or, as an explicit choice, held to factor x the anchor (still checked).
    assert ssc.tightened_nmse_bound(5.25e-4, anchor_relative=True) == pytest.approx(1.575e-3)
    # A measured-zero anchor NMSE must not produce a zero bound.
    assert ssc.tightened_nmse_bound(0.0) == ssc.SERVED_SHAPE_NMSE_FLOOR
    # Review of bb169f4b: an anchor the old policy accepted keeps its old bound.
    assert ssc.tightened_nmse_bound(5e-5, factor=3.0) == pytest.approx(1e-4)


def _old_bound(anchor, factor=3.0):
    """The pre-bb169f4b policy (cap 1e-4), verbatim."""
    return min(1e-4, max(ssc.SERVED_SHAPE_NMSE_FLOOR, anchor * factor))


def test_every_anchor_the_old_policy_accepted_keeps_its_old_bound():
    import numpy as np
    for anchor in np.concatenate(([0.0], np.logspace(-12, np.log10(9.9999e-5), 400))):
        assert ssc.tightened_nmse_bound(float(anchor)) == _old_bound(float(anchor)), anchor
    assert ssc.SERVED_SHAPE_NMSE_LEGACY_CAP == 1e-4


def test_the_real_ds41_record_keeps_every_old_bound(capsys):
    """Read-only replay of the first real calibration record (DS41)."""
    import json
    record = Path("/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/"
                  "store-b0ba1d427/served_shape/calibration-20261006T104207Z.json")
    if not record.is_file():
        pytest.skip("DS41 calibration record not present")
    values = [row["nmse"] for row in json.loads(record.read_text())["measurements"]]
    regimes = {"old (<1e-4)": 0, "new [1e-4,5e-4)": 0, "anchor>=5e-4": 0}
    for anchor in values:
        if anchor < 1e-4:
            regimes["old (<1e-4)"] += 1
            assert ssc.tightened_nmse_bound(anchor) == _old_bound(anchor)
        elif anchor < 5e-4:
            regimes["new [1e-4,5e-4)"] += 1
            bound = ssc.tightened_nmse_bound(anchor)
            assert anchor < bound <= 5e-4
        else:
            regimes["anchor>=5e-4"] += 1
            with pytest.raises(ValueError):
                ssc.tightened_nmse_bound(anchor)
    print(f"DS41 record regimes: {regimes}")
    assert sum(regimes.values()) == len(values)


def test_anchor_relative_cases_are_recorded_and_only_they_may_exceed_the_cap(tmp_path):
    import json
    anchor = dict(ANCHOR_NMSE)
    key = ("ds41_shexp_down", "IQ3_XXS", 5)
    assert key in anchor
    anchor[key] = 5.25e-4
    with pytest.raises(ValueError):
        ssc.case_set(anchor)
    over = ssc.anchor_exceeds_generic(anchor)
    assert over == {key}
    cases = ssc.case_set(anchor, anchor_relative_keys=over)
    bound = next(c.max_nmse for c in cases if (c.shape.name, c.type_a, c.n) == key)
    assert bound == pytest.approx(1.575e-3)
    path = tmp_path / "manifest.json"
    ssc.write_manifest(path, cases, anchor_relative_keys=over)
    assert ssc.load_manifest(path) == cases
    body = json.loads(path.read_text())
    body["anchor_relative"] = []
    path.write_text(json.dumps(body))
    with pytest.raises(ssc.ManifestRefused):
        ssc.load_manifest(path)   # an unrecorded case above the cap refuses


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
    # round-14: only VALID pairs -- the type's block size must divide the shape's k
    valid = sum(1 for shape in ssc.SERVED_SHAPES for t in ssc.WITNESS_TYPES
                if shape.k % ssc.BLOCK_SIZE[t] == 0)
    assert len(cases) == valid * len(ssc.SERVED_WIDTHS)
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
    assert ssc.binary_has_case_set(build) is False   # round-12: thread control required
    binary.write_bytes(b"\0ELF" + ssc.CASE_SET_ID.encode() + b"\0"
                       + ssc.BACKEND_THREADS_ENV.encode())
    assert ssc.binary_has_case_set(build) is False   # round-17: seed marker required
    binary.write_bytes(b"\0ELF" + ssc.CASE_SET_ID.encode() + b"\0"
                       + ssc.BACKEND_THREADS_ENV.encode() + ssc.SEED_MARKER.encode())
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
        tmp_path / "build", resolved_recipe=_recipe(), manifest_path=tmp_path / "absent.json",
        lane="ds41")
    assert verdict.passed is False
    assert verdict.gate == "oracle_unavailable"   # round-12: unavailable, not numerical
    assert "not available" in verdict.reason


def test_check_served_shape_case_set_fails_closed_on_missing_binary_literal(tmp_path):
    cases = ssc.case_set(ANCHOR_NMSE, lane="ds41")
    manifest = tmp_path / "manifest.json"
    ssc.write_manifest(manifest, cases, lane="ds41")
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "test-backend-ops").write_bytes(b"\0ELF\0")
    verdict = gates.check_served_shape_case_set(
        build, resolved_recipe=_recipe(), manifest_path=manifest, lane="ds41")
    assert verdict.passed is False
    assert "before deterministic" in verdict.reason   # round-17: seed marker first


def test_check_served_shape_case_set_runs_op_correctness_when_available(tmp_path, monkeypatch):
    cases = ssc.case_set(ANCHOR_NMSE, lane="ds41")
    manifest = tmp_path / "manifest.json"
    ssc.write_manifest(manifest, cases, lane="ds41")
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "test-backend-ops").write_bytes(
        b"\0ELF" + ssc.CASE_SET_ID.encode() + b"\0" + ssc.BACKEND_THREADS_ENV.encode()
        + ssc.SEED_MARKER.encode())
    calls = []

    def fake_op_correctness(build_dir, *, op, backend, resolved_recipe, params_filter,
                            environment_overrides, expected_cases):
        calls.append(dict(op=op, params_filter=params_filter,
                          environment_overrides=environment_overrides,
                          expected_cases=expected_cases))
        return gates.Verdict("correctness", True, "ok")

    monkeypatch.setattr(gates, "op_correctness", fake_op_correctness)
    unknown = gates.check_served_shape_case_set(
        build, resolved_recipe=_recipe(), manifest_path=manifest, lane="ds41")
    assert unknown.passed is False and unknown.gate == "oracle_unavailable"
    assert "thread count" in unknown.reason
    no_lane = gates.check_served_shape_case_set(
        build, resolved_recipe=_served_recipe(), manifest_path=manifest)
    assert no_lane.passed is False and no_lane.gate == "oracle_unavailable"
    verdict = gates.check_served_shape_case_set(
        build, resolved_recipe=_served_recipe(), manifest_path=manifest, lane="ds41")
    assert verdict.passed is True
    # 2026-10-06: sharded -- one op_correctness call per shard (default up to
    # SERVED_SHAPE_CASE_SET_DEFAULT_SHARDS), merged. Every case is requested exactly
    # once across the shards, and every shard shares the same op/env overrides.
    assert len(calls) == min(gates.SERVED_SHAPE_CASE_SET_DEFAULT_SHARDS, len(cases))
    assert sum(call["expected_cases"] for call in calls) == len(cases)
    for call in calls:
        assert call["environment_overrides"] == ((ssc.CASE_SET_ENV, ssc.CASE_SET_ID),
                                                  (ssc.BACKEND_THREADS_ENV, "48"))
        assert "MUL_MAT" in call["op"] and "MUL_MAT_ID" in call["op"]


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

    def failing_served_shape_check(build, *, resolved_recipe, manifest_path, **_kw):
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

    def passing_served_shape_check(build, *, resolved_recipe, manifest_path, **_kw):
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


def test_each_lanes_routed_corpus_uses_its_served_expert_count_and_dispersed_ids():
    """Round-12/13: the bundle tier runs each lane's SERVED expert count / used with
    deterministic ids spanning the whole range."""
    q = ssc.routed_shapes("q38fn")
    assert all(s.n_mats == 512 and s.n_used == 10 and s.routed for s in q)
    assert {(s.k, s.m) for s in q} == {(2560, 640), (640, 2560)}
    assert set(ssc.routed_types("q38fn")) == {"IQ3_S", "IQ4_NL", "IQ4_XS", "Q8_0", "Q6_K"}
    d = ssc.routed_shapes("ds41")
    assert all(s.n_mats == 384 and s.n_used == 6 for s in d)
    assert ssc.ROUTED_WIDTHS == (2, 3, 4, 5)
    for n_mats, n_used in ((512, 10), (384, 6)):
        rows = [ssc.route_ids(n_mats, n_used, r) for r in range(5)]
        assert all(len(set(row)) == n_used for row in rows)
        ids = {i for row in rows for i in row}
        assert len(ids) == 5 * n_used and max(ids) == n_mats - 1
        assert any(i >= n_mats * 3 // 4 for i in ids) and any(i < n_mats // 4 for i in ids)
    q_ids = {i for r in range(5) for i in ssc.route_ids(512, 10, r)}
    assert any(256 <= i < 384 for i in q_ids) and any(i >= 384 for i in q_ids)
    anchor = {(t[0].name, t[1], t[2]): 1e-6 for t in ssc.calibration_triples("q38fn")}
    routed = ssc.case_set(anchor, routed=True, lane="q38fn")
    # gate/up IQ3_S, IQ4_XS (+Q6_K extra); down (k=640) IQ4_NL, Q8_0 only
    assert len(routed) == (3 + 2) * 4
    pairs = {(c.shape.name.split("_routed")[0], c.type_a) for c in routed}
    assert pairs == {("q38fn_expert_gate_up", "IQ3_S"), ("q38fn_expert_gate_up", "IQ4_XS"),
                     ("q38fn_expert_gate_up", "Q6_K"), ("q38fn_expert_down", "IQ4_NL"),
                     ("q38fn_expert_down", "Q8_0")}
    assert routed[0].vars().endswith(",routed=1,max_nmse=3e-06")
    block = ssc.backend_ops_patch_block(ssc.case_set(anchor, lane="q38fn"), routed)
    assert "test_mul_mat_id_served_routed(" in block and ssc.ROUTED_CASE_SET_ID in block
    assert "const int64_t step = (n_mats / n_used) | 1;" in block
    assert "static int autokernel_backend_threads()" in block
    with pytest.raises(ValueError):
        ssc.case_set(anchor, routed=True)   # no lane: refused


def test_the_model_selects_its_lane():
    assert ssc.lane_for_model(ssc.LANE_PROFILES["ds41"].model) == "ds41"
    shard2 = ssc.LANE_PROFILES["q38fn"].model.replace("00001-of", "00002-of")
    assert ssc.lane_for_model(shard2) == "q38fn"
    assert ssc.lane_for_model("/elsewhere/model.gguf") is None


def test_routed_manifest_round_trips_separately(tmp_path):
    anchor = {(t[0].name, t[1], t[2]): 1e-6 for t in ssc.calibration_triples("q38fn")}
    routed = ssc.case_set(anchor, routed=True, lane="q38fn")
    path = tmp_path / "manifest-routed.json"
    ssc.write_manifest(path, routed, routed=True, lane="q38fn")
    assert ssc.load_manifest(path, routed=True) == routed
    assert ssc.load_manifest(path, routed=True, lane="q38fn") == routed
    with pytest.raises(ssc.ManifestRefused):
        ssc.load_manifest(path, routed=True, lane="ds41")
    with pytest.raises(ssc.ManifestRefused):
        ssc.load_manifest(path)


def test_the_patch_sets_the_backend_thread_count_the_tool_uses(tmp_path):
    test_file = tmp_path / "test-backend-ops.cpp"
    test_file.write_text(
        "static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {\n"
        "    std::vector<std::unique_ptr<test_case>> test_cases;\n    return test_cases;\n}\n"
        "int main() {\n"
        "            ggml_backend_set_n_threads_fn(backend.get(), N_THREADS);\n}\n")
    ssc.apply_patch_block(test_file, ssc.calibration_patch_block("q38fn"))
    ssc.apply_patch_block(test_file, ssc.calibration_patch_block("q38fn"))
    text = test_file.read_text()
    assert text.count(ssc.THREADS_PATCHED) == 1 and "N_THREADS);" not in text.split("int main")[1]


def test_the_calibration_set_covers_both_corpora():
    for lane in ssc.LANE_PROFILES:
        assert len(ssc.calibration_triples(lane)) == (
            len(ssc.canonical_triples(lane=lane))
            + len(ssc.canonical_triples(routed=True, lane=lane)))


def test_shard_sequence_is_disjoint_and_covers_every_item_exactly_once():
    items = tuple(range(37))
    for n in (1, 2, 5, 16, 37, 1000):
        shards = ssc.shard_sequence(items, n)
        flat = [x for shard in shards for x in shard]
        assert sorted(flat) == sorted(items)          # every item exactly once
        assert len(flat) == len(set(flat)) == len(items)
        assert all(shard for shard in shards)          # no empty shard
        assert len(shards) == min(n, len(items))       # clamped, never padded with empties
    assert ssc.shard_sequence((), 5) == ()
    with pytest.raises(ValueError):
        ssc.shard_sequence(items, 0)


def test_parse_calibration_restricts_to_the_given_shard_and_rejects_leakage():
    triples = ssc.calibration_triples("q38fn")
    shards = ssc.shard_sequence(triples, 4)
    shard0 = shards[0]
    lines = [f"{ssc.CALIBRATION_MARKER}\t{ssc.calibration_vars(*t)}\t1e-6" for t in shard0]
    measured = ssc.parse_calibration("\n".join(lines), "q38fn", triples=shard0)
    assert len(measured) == len(shard0)
    # a case from ANOTHER shard leaking into this shard's output is an unexpected line.
    foreign = shards[1][0]
    leaked = lines + [f"{ssc.CALIBRATION_MARKER}\t{ssc.calibration_vars(*foreign)}\t1e-6"]
    with pytest.raises(ValueError, match="unexpected calibration line"):
        ssc.parse_calibration("\n".join(leaked), "q38fn", triples=shard0)
    # a missing case in this shard's own assignment is caught too.
    with pytest.raises(ValueError, match="missing"):
        ssc.parse_calibration("\n".join(lines[:-1]), "q38fn", triples=shard0)



@pytest.mark.parametrize("lane", sorted(ssc.LANE_PROFILES))
def test_every_generated_case_respects_the_block_size(lane):
    """Round-14: Q38FN's down shape has k=640; no 256-block type may be paired with it."""
    for shape, type_a, _n in ssc.calibration_triples(lane):
        assert shape.k % ssc.BLOCK_SIZE[type_a] == 0, (shape.name, type_a)
    down = [t for s, t, _ in ssc.canonical_triples() if s.name == "q38fn_expert_down"]
    assert down and set(down) <= {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL"}
    assert set(ssc.BLOCK_SIZE) == set(ssc.WITNESS_TYPES)


def test_routed_types_follow_the_served_direction():
    gate, down = ssc.routed_shapes("q38fn")
    assert ssc.routed_types("q38fn", gate) == ("Q6_K", "IQ4_XS", "IQ3_S")
    assert ssc.routed_types("q38fn", down) == ("Q8_0", "IQ4_NL")



def test_the_candidate_corpus_is_lane_partitioned():
    """Round-15: each lane measures only its own served shapes (+ identical-dim shapes
    of another lane, none today)."""
    for lane in ssc.LANE_PROFILES:
        names = {shape.name for shape in ssc.lane_served_shapes(lane)}
        assert names and all(name.startswith(lane + "_") for name in names)
    assert len(ssc.canonical_triples(lane="ds41")) == 340
    assert len(ssc.canonical_triples(lane="q38fn")) == 230
    assert len(ssc.calibration_triples("ds41")) == 348
    assert len(ssc.calibration_triples("q38fn")) == 250


def test_manifests_record_seed_scheme_and_partition(tmp_path):
    import json
    anchor = {(t[0].name, t[1], t[2]): 1e-6 for t in ssc.calibration_triples("q38fn")}
    cases = ssc.case_set(anchor, lane="q38fn")
    path = tmp_path / "manifest.json"
    ssc.write_manifest(path, cases, lane="q38fn")
    body = json.loads(path.read_text())
    assert body["seed_scheme"] == ssc.SEED_SCHEME and body["lane"] == "q38fn"
    assert body["partition"] == [s.name for s in ssc.lane_served_shapes("q38fn")]
    assert ssc.load_manifest(path, lane="q38fn") == cases
    with pytest.raises(ssc.ManifestRefused):
        ssc.load_manifest(path, lane="ds41")
    for field, value in (("seed_scheme", "random-inputs"), ("partition", ["x"])):
        tampered = dict(body, **{field: value})
        path.write_text(json.dumps(tampered))
        with pytest.raises(ssc.ManifestRefused):
            ssc.load_manifest(path, lane="q38fn")


def test_generated_cpp_seeds_every_case_from_its_case_key_hash():
    """Round-15: deterministic per-case inputs, identical for calibration and gating."""
    anchor = {(t[0].name, t[1], t[2]): 1e-6 for t in ssc.calibration_triples("q38fn")}
    final = ssc.backend_ops_patch_block(ssc.case_set(anchor, lane="q38fn"),
                                        ssc.case_set(anchor, routed=True, lane="q38fn"))
    calib = ssc.calibration_patch_block("q38fn")
    for block in (final, calib):
        assert "suite_seed_begin(0x414b53455256ULL, (size_t) h, \"AK_SERVED_SHAPE\");" in block
        assert "h ^= c; h *= 0x100000001B3ULL;" in block and "0x14650FB0739D0383ULL" in block
        # every subclass seeds BEFORE the base initialisation (covers the MUL_MAT_ID ids)
        assert block.count("autokernel_seed_case(test_mul_mat::vars());") == 1
        assert block.count("autokernel_seed_case(test_mul_mat_id::vars());") == 1
        assert block.count('autokernel_seed_case(test_mul_mat_id::vars() + ",routed=1");') == 1
        for match in re.finditer(r"autokernel_seed_case\(([^;]*)\);\n\s*(\S+)", block):
            assert "initialize_tensors(ctx)" in match.group(2), match.group(0)
    assert ssc.AK_SERVED_SHAPE_SEED == 0x414B53455256


def test_the_same_case_key_and_seed_for_the_bound_and_calibrate_variants():
    anchor = {(t[0].name, t[1], t[2]): 1e-6 for t in ssc.calibration_triples("q38fn")}
    cases = (ssc.case_set(anchor, lane="q38fn")
             + ssc.case_set(anchor, routed=True, lane="q38fn"))
    seeds = set()
    for case in cases:
        key = ssc.case_key(case.shape, case.type_a, case.n)
        bound_key = case.vars()[:case.vars().rindex(",max_nmse=")]
        calib = ssc.calibration_vars(case.shape, case.type_a, case.n)
        assert key == bound_key == calib[:-len(",calibrate=1")]
        seeds.add(ssc.case_seed_index(key))
    assert len(seeds) == len(cases)    # distinct inputs per case
    # pinned value (the tree's suite_seed_hash_string offset basis, see case_seed_index)
    assert ssc.case_seed_index("abc") == 0xE16801510DB89EFD



@pytest.mark.parametrize("routed", [False, True])
def test_the_gate_refuses_a_test_backend_ops_built_before_seeding(tmp_path, routed):
    """Round-17: both gate checks (candidate and routed corpus) refuse a stale binary
    before running or judging anything."""
    cases = ssc.case_set(ANCHOR_NMSE, routed=routed, lane="ds41") if not routed else None
    manifest = tmp_path / "manifest.json"
    if routed:
        anchor = {(t[0].name, t[1], t[2]): 1e-6 for t in ssc.calibration_triples("ds41")}
        cases = ssc.case_set(anchor, routed=True, lane="ds41")
    ssc.write_manifest(manifest, cases, routed=routed, lane="ds41")
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    set_id = ssc.ROUTED_CASE_SET_ID if routed else ssc.CASE_SET_ID
    (build / "bin" / "test-backend-ops").write_bytes(
        b"\0ELF" + set_id.encode() + b"\0" + ssc.BACKEND_THREADS_ENV.encode())
    called = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(gates, "op_correctness", lambda *a, **k: called.append(1))
    try:
        verdict = gates.check_served_shape_case_set(
            build, resolved_recipe=_served_recipe(), manifest_path=manifest,
            routed=routed, lane="ds41")
    finally:
        monkey.undo()
    assert verdict.passed is False and verdict.gate == "oracle_unavailable"
    assert "before deterministic" in verdict.reason and not called


def test_generated_cpp_carries_the_seed_marker_and_the_input_hash_mode():
    anchor = {(t[0].name, t[1], t[2]): 1e-6 for t in ssc.calibration_triples("q38fn")}
    final = ssc.backend_ops_patch_block(ssc.case_set(anchor, lane="q38fn"),
                                        ssc.case_set(anchor, routed=True, lane="q38fn"))
    for block in (final, ssc.calibration_patch_block("q38fn")):
        assert f'"{ssc.SEED_MARKER}"' in block
        assert f'std::getenv("{ssc.INPUT_HASH_ENV}")' in block
        assert f'"{ssc.INPUT_HASH_MARKER}\\t%s\\t%016llx\\n"' in block
        # each subclass hashes its inputs AFTER initialisation (and after the routed ids)
        for body in re.findall(r"void initialize_tensors\(ggml_context \* ctx\) override "
                               r"\{(.*?)\n    \}", block, re.S):
            stmts = [s.strip() for s in body.strip().splitlines()]
            assert stmts[0].startswith("autokernel_seed_case(")
            assert stmts[-1].startswith("autokernel_report_inputs(ctx, ")
            assert stmts[0][len("autokernel_seed_case("):] == \
                stmts[-1][len("autokernel_report_inputs(ctx, "):]
        assert block.count("autokernel_report_inputs(ctx, ") == 3


def test_input_hashes_compare_by_case_key_regardless_of_order():
    """Round-17 compiled-level check, stubbed: a calibration run and a gate run print
    `AK_INPUT_HASH` per case in different orders; comparison is by key."""
    calib = "\n".join(["noise", f"{ssc.INPUT_HASH_MARKER}\tk1\taaaa",
                       f"{ssc.INPUT_HASH_MARKER}\tk2\tbbbb", "AK_SERVED_NMSE\tk1\t1e-6"])
    gate = "\n".join([f"{ssc.INPUT_HASH_MARKER}\tk2\tbbbb", "  MUL_MAT(...): OK",
                      f"{ssc.INPUT_HASH_MARKER}\tk1\taaaa"])
    assert ssc.input_hash_mismatches(ssc.parse_input_hashes(calib),
                                     ssc.parse_input_hashes(gate)) == []
    drift = gate.replace("k1\taaaa", "k1\tcccc")
    assert ssc.input_hash_mismatches(ssc.parse_input_hashes(calib),
                                     ssc.parse_input_hashes(drift)) == ["k1"]
    assert ssc.input_hash_mismatches({"k1": "a"}, {}) == ["k1"]
    with pytest.raises(ValueError):
        ssc.parse_input_hashes(calib + f"\n{ssc.INPUT_HASH_MARKER}\tk1\tdddd")
