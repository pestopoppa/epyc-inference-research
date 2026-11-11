"""INF-45 E5 Stage-B v2 grid: the five binding requirements, pools, attestation, tagging."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import e5_cell_manifests as e5  # noqa: E402
import e5_v2_stageb_grid as g  # noqa: E402
import e5_v2_qualify_banked as q  # noqa: E402

OUT = g.DEFAULT_OUT


def _groups():
    return g.split_groups(g.build_groups(g.load_depth_blocks(OUT)))


def test_grid_validates_and_covers_requirement_a():
    groups = _groups()
    assert g.validate_all(groups) == []
    for mk in g.MODEL_GROUPS:
        cells = [c for cs in groups.values() for c in cs if c["model_key"] == mk and "depth_probe" not in c]
        full = [c for c in cells if c["instances"][0]["cpu_list"] == "0-95"]
        assert {c["np"] for c in full} == set(g.C1_NP)
        assert all(c["instances"][0]["numactl_policy"] == "interleave=all" for c in full)


def test_requirement_b_straddle_needs_interleave_or_declared_control():
    cell = g.make_v2_cell("qwen36_q8_0", "C1b", 4)
    assert g.placement_violations(cell) == []
    bad = copy.deepcopy(cell)
    for inst in bad["instances"]:
        inst["numactl_policy"] = "none"
    assert any("straddles" in v for v in g.placement_violations(bad))
    bad.pop("defect_replication_control")
    assert any("never implicit" in v for v in g.placement_violations(bad))
    ctl = g.make_v2_cell("qwen36_q8_0", "C1b-defect", 4)
    assert ctl["defect_replication_control"] is True and g.placement_violations(ctl) == []


def test_requirement_d_policy_must_cover_cpuset_nodes():
    cell = g.make_v2_cell("qwen36_q8_0", "C1b", 4)
    cell["instances"][0]["numactl_policy"] = "interleave=2,3"  # half0 lives on N0,N1
    assert any("mismatch" in v for v in g.placement_violations(cell))


def test_requirement_c_every_cell_resets_per_cell_under_placement_protocol():
    for cells in _groups().values():
        for c in cells:
            assert c["cache_reset"] == "per_cell"
            assert c["placement_protocol"] == "P-BENCH-PLACEMENT-1"


def test_one_model_per_group_and_groups_fit_the_grant():
    for name, cells in _groups().items():
        assert len({c["model_key"] for c in cells}) == 1, name
        assert sum(g.cell_estimate_s(c) for c in cells) <= g.MAX_GROUP_S, name


def test_depth_points_beyond_16k_exist_for_both_models():
    groups = _groups()
    for mk in g.MODEL_GROUPS:
        depths = {c["depth_probe"]["target_depth_tokens"] for cs in groups.values() for c in cs
                  if c["model_key"] == mk and "depth_probe" in c}
        assert max(depths) > 16384


def test_depth_pool_is_deterministic_and_pinned(tmp_path):
    src = tmp_path / "pool.jsonl"
    src.write_text("".join(json.dumps({"id": f"q{i}", "prompt": "x" * 400 + str(i)}) + "\n" for i in range(50)))
    a = g.build_depth_pool(2048, src, n_prompts=3)
    b = g.build_depth_pool(2048, src, n_prompts=3)
    assert a == b and len(a) == 3
    assert all(len(r["prompt"]) >= int(2048 * g.CHARS_PER_TOKEN_EST) for r in a)
    for t, blk in g.load_depth_blocks(OUT).items():
        assert g.verified_pool(OUT, blk).exists()


def test_depth_probe_opt_in_is_validated_by_schema_owner():
    cell = g.make_v2_cell("qwen36_q8_0", "C1", 2, g.load_depth_blocks(OUT)[32768])
    assert e5.validate_cell_manifest(cell) == []
    broken = copy.deepcopy(cell)
    broken["ctx"] = 8192
    assert any("depth_probe.per_stream_ctx*np" in v for v in e5.validate_cell_manifest(broken))
    plain = g.make_v2_cell("qwen36_q8_0", "C1", 2)
    plain["prompt_caps"]["max_prompt_chars"] = 50000
    assert e5.validate_cell_manifest(plain)  # no depth_probe -> the old 4096 cap still binds


def test_attest_instance_fails_closed():
    assert g.attest_instance("interleave=all", "0-95", None, False)[0] == "FAIL"
    even = {"N0": 100, "N1": 100, "N2": 100, "N3": 100}
    assert g.attest_instance("interleave=all", "0-95", even, False)[0] == "PASS"
    assert g.attest_instance("interleave=0,1,2,3", "0-47,96-143", even, False)[0] == "PASS"
    warm = {"N0": 390, "N1": 10, "N2": 0, "N3": 0}
    assert g.attest_instance("interleave=all", "0-95", warm, False)[0] == "FAIL"
    assert g.attest_instance("none", "0-23,96-119", {"N0": 100}, False)[0] == "PASS"
    assert g.attest_instance("none", "0-47,96-143", even, True)[0] == "DEFECT-CONTROL"


def test_attest_run_requires_cache_witness(tmp_path):
    (tmp_path / "affinity").mkdir()
    row = {"cell_id": "c", "instances": [{"cpu_list": "0-95", "numactl_policy": "interleave=all"}],
           "cache_reset_method": None}
    (tmp_path / "cells.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "affinity" / "c.json").write_text(json.dumps(
        {"instances": [{"memory_placement": {"pages_by_node": {"N0": 1, "N1": 1, "N2": 1, "N3": 1}}}]}))
    assert g.attest_run(tmp_path)[0]["policy_attestation"] == "FAIL"
    row["cache_reset_method"] = "drop_caches"
    (tmp_path / "cells.jsonl").write_text(json.dumps(row) + "\n")
    assert g.attest_run(tmp_path)[0]["policy_attestation"] == "PASS"


def test_qualify_tags(tmp_path):
    root = tmp_path
    live = root / "data/batched_decode/e5_manifests/m"
    live.mkdir(parents=True)
    w0 = {"cell_id": "a", "model_key": "m", "window": "W0", "reasoning": "auto", "architecture": "qwen35moe",
          "instances": [{"cpu_list": "0-23,96-119", "numactl_policy": "none"}]}
    gem = {"cell_id": "b", "model_key": "m", "window": "W2", "architecture": "gemma4",
           "instances": [{"cpu_list": "0-95", "numactl_policy": "interleave=all"}]}
    half = {"cell_id": "c", "model_key": "m", "window": "W1", "reasoning": "auto", "architecture": "qwen35moe",
            "instances": [{"cpu_list": "0-47,96-143", "numactl_policy": "none"}]}
    for name, m in (("a", w0), ("b", gem), ("c", half)):
        (live / f"{name}.json").write_text(json.dumps(m))
    t = {x["cell_id"]: x for x in (q.tag_cell(live / f"{n}.json", m, root) for n, m in
                                     (("a", w0), ("b", gem), ("c", half)))}
    assert t["a"]["w0_mismatch"] and t["a"]["verdict"] == "DISCARD"
    assert t["b"]["reasoning_axis"] == "DISCARD"
    assert t["c"]["placement_axis"] == "DISCARD"
    assert all(x["cache_axis"] == "UNRESOLVED" for x in t.values())
    out = root / "tags.jsonl"
    assert q.main(["--root", str(root), "--output", str(out), "--stamp"]) == 0
    assert "superseded_by_run" in json.loads((live / "a.json").read_text())
    assert "superseded_by_run" not in json.loads((live / "c.json").read_text())
