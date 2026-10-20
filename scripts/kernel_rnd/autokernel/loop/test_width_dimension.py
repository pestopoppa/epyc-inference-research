"""AKX-ALL-27: observe-only width-invariance keep dimension."""
import copy
import json
from pathlib import Path

import pytest

from . import surface_validation as sv
from . import width_dimension as wd

REPO = Path(__file__).resolve().parents[4]
CPU_CELLS = REPO / "artifacts" / "akx_all_26_cpu_20261020" / "cells.jsonl.gz"
HIP_CELLS = Path("/mnt/raid0/llm/worktrees/akx-all-27-root-20261020/artifacts/gpu-queue-20261020/"
                 "akx-all-26/rocm0_cells.jsonl")


def T(**cells):
    return {k: {"status": v[0], "first_fail_k": v[1]} for k, v in cells.items()}


def test_touched_families_only_width_sensitive_sources():
    hits = wd.touched_families(["ggml/src/ggml-cuda/mmvq.cu", "ggml/src/ggml-cuda/fattn-vec.cuh",
                                "ggml/src/ggml-cuda/gated_delta_net.cu", "ggml/src/ggml-cpu/ops.cpp"])
    assert set(hits) == {"mmvq", "fattn", "gdn"}
    assert wd.touched_families(["ggml/src/ggml-cuda/norm.cu"]) == {}


def test_compare_is_relative_not_absolute():
    anchor = T(a=("FAIL", 2), b=("PASS", None), c=("FAIL", 5), d=("FAIL", 3), e=("PASS", None))
    same = wd.compare_tables(anchor, copy.deepcopy(anchor))
    assert same["verdict"] == "unchanged" and same["coverage_complete"]
    keep = T(a=("FAIL", 2), b=("FAIL", 4), c=("FAIL", 3), d=("PASS", None), e=("PASS", None))
    r = wd.compare_tables(anchor, keep)
    assert r["verdict"] == "mixed"
    assert [x["cell"] for x in r["new_failing"]] == ["b"]
    assert r["lower_first_fail_k"] == [{"cell": "c", "anchor": 5, "keep": 3}]
    assert [x["cell"] for x in r["fixed"]] == ["d"]
    assert r["unchanged_fail"] == ["a"] and r["unchanged_pass"] == ["e"]


def test_higher_first_fail_alone_is_improved():
    r = wd.compare_tables(T(a=("FAIL", 2)), T(a=("FAIL", 6)))
    assert r["verdict"] == "improved"


def test_unsupported_err_void_and_missing_cells_are_listed_never_pass():
    anchor = T(a=("PASS", None), b=("UNSUPPORTED", None), c=("PASS", None), d=("PASS", None))
    keep = T(a=("ERR", None), b=("UNSUPPORTED", None), c=("VOID_AA", None))
    r = wd.compare_tables(anchor, keep)
    assert {x["cell"] for x in r["indeterminate"]} == {"a", "c"}
    assert [x["cell"] for x in r["missing_in_keep"]] == ["d"]
    assert r["coverage_complete"] is False and r["verdict"] == "unchanged"
    assert all(x["cell"] != "b" for x in r["indeterminate"])


def test_dimension_dispositions_and_observe_only_flags():
    fp = ["ggml/src/ggml-cuda/mmvq.cu"]
    na = wd.width_dimension(footprint=["x.cu"], anchor_table=T(a=("PASS", None)), keep_table=None)
    assert na["disposition"] == "not_applicable" and na["refuses"] is False
    nm = wd.width_dimension(footprint=fp, anchor_table=T(a=("PASS", None)), keep_table=None)
    assert nm["disposition"] == "not_measured" and "probe did not run" in nm["reason"]
    nm2 = wd.width_dimension(footprint=fp, anchor_table=None, keep_table=T(a=("PASS", None)))
    assert nm2["disposition"] == "not_measured"
    m = wd.width_dimension(footprint=fp, anchor_table=T(a=("PASS", None)), keep_table=T(a=("FAIL", 2)))
    assert m["disposition"] == "measured" and m["comparison"]["verdict"] == "worse"
    assert m["observe_only"] is True and m["refuses"] is False
    assert m["model_level"]["disposition"] == "not_measured"


def _wr(flips=0, nt1=10, nt2=3):
    return {"near_tie_lt_1_0": nt1, "near_tie_lt_0_25": nt2, "top1_flips": flips,
            "row_mean_centred_max_delta": 0.01, "max_dP": 0.002}


CORPUS = {"kind": "real_code_near_tie_dense", "name": "code-nt-v1"}


def test_model_level_dispositions():
    full = {w: _wr() for w in wd.MODEL_WIDTHS}
    assert wd.model_level_record(None, corpus=CORPUS)["disposition"] == "not_measured"
    assert wd.model_level_record(full, corpus=CORPUS)["disposition"] == "zero_flips_informative"
    flips = {**full, 5: _wr(flips=2)}
    assert wd.model_level_record(flips, corpus=CORPUS)["disposition"] == "flips_observed"
    confident = {w: _wr(nt1=0, nt2=0) for w in wd.MODEL_WIDTHS}
    assert wd.model_level_record(confident, corpus=CORPUS)["disposition"] == "uninformative"
    assert wd.model_level_record(full, corpus={"kind": "repetitive_text", "name": "x"}
                                 )["disposition"] == "corpus_not_admissible"
    assert wd.model_level_record({2: _wr()}, corpus=CORPUS)["disposition"] == "incomplete"
    with pytest.raises(ValueError, match="perplexity"):
        wd.model_level_record({w: dict(_wr(), ppl=5.1) for w in wd.MODEL_WIDTHS}, corpus=CORPUS)


def test_keep_dimensions_gate_unchanged_by_width_record():
    base = sv.keep_dimensions(declared=("short_decode",), primary="short_decode", comparisons={},
                              capacity=None)
    assert "width_invariance" not in base
    worse = wd.width_dimension(footprint=["mmvq.cu"], anchor_table=T(a=("PASS", None)),
                               keep_table=T(a=("FAIL", 2)))
    carried = sv.keep_dimensions(declared=("short_decode",), primary="short_decode", comparisons={},
                                 capacity=None, width_invariance=worse)
    assert carried["width_invariance"]["comparison"]["verdict"] == "worse"
    assert {k: v for k, v in carried.items() if k != "width_invariance"} == base
    assert carried["passed"] is True and set(carried["dimensions"]) == set(sv.KEEP_DIMENSIONS)


def test_load_table_rejects_empty(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("\n")
    with pytest.raises(ValueError):
        wd.load_table(p)


@pytest.mark.skipif(not CPU_CELLS.exists(), reason="AKX-ALL-26 CPU evidence not in tree")
def test_cpu_table_self_compare_is_unchanged_and_matches_recorded_findings():
    t = wd.load_table(CPU_CELLS)
    mm = {k: v for k, v in t.items() if k.startswith("cpu|cpu|24|MUL_MAT|Q8_0|")}
    assert mm and all(v["status"] == "FAIL" and v["first_fail_k"] == 2 for v in mm.values())
    r = wd.compare_tables(t, t)
    assert r["verdict"] == "unchanged" and r["counts"]["new_failing"] == 0
    # perturb one passing cell: exactly that cell is reported
    key = next(k for k, v in t.items() if v["status"] == "PASS")
    keep = copy.deepcopy(t)
    keep[key] = {"status": "FAIL", "first_fail_k": 3}
    r2 = wd.compare_tables(t, keep)
    assert r2["verdict"] == "worse" and [x["cell"] for x in r2["new_failing"]] == [key]


@pytest.mark.skipif(not HIP_CELLS.exists(), reason="AKX-ALL-26 HIP evidence not in tree")
def test_hip_table_loads_cells():
    t = wd.load_table(HIP_CELLS)
    from collections import Counter
    assert len(t) == 116  # 116 `cell` records; (backend, id, sub) keys are unique
    assert Counter(v["status"] for v in t.values()) == {"FAIL": 97, "UNSUPPORTED": 17, "PASS": 2}
    assert wd.compare_tables(t, t)["verdict"] == "unchanged"
