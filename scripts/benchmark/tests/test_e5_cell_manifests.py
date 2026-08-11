#!/usr/bin/env python3
"""Offline tests for the E5 cell-manifest schema + pre-registered grid generator.

Runs under pytest if available, and — because the research repo ships no pytest —
also stands alone via the stdlib runner in ``__main__``:

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        scripts/benchmark/tests/test_e5_cell_manifests.py

Every test is fully OFFLINE and inference-free: pure schema validation and
grid-generation checks on in-memory dicts plus tempdir CLI round-trips. No
port, process, or server is ever touched.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _BENCHMARK_DIR.parent
_RESEARCH_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_RESEARCH_ROOT), str(_SCRIPTS_DIR), str(_BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e5_cell_manifests as e5  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (stdlib only, so they work under both runners)
# ---------------------------------------------------------------------------

_GRID_CACHE: list[dict] | None = None


def grid() -> list[dict]:
    global _GRID_CACHE
    if _GRID_CACHE is None:
        _GRID_CACHE = e5.build_grid()
    return _GRID_CACHE


def by_id(cell_id: str) -> dict:
    for cell in grid():
        if cell["cell_id"] == cell_id:
            return copy.deepcopy(cell)
    raise AssertionError(f"cell not found in grid: {cell_id}")


def base_manifest() -> dict:
    """A known-valid Stage-B manifest for mutation tests."""
    return by_id("qwen36_q8_0-C3-np8")


def assert_violation(violations: list[str], needle: str) -> None:
    assert any(needle in v for v in violations), (
        f"expected a violation containing {needle!r}, got: {violations}"
    )


def assert_valid(manifest: dict) -> None:
    violations = e5.validate_cell_manifest(manifest)
    assert violations == [], f"expected valid manifest, got: {violations}"


# ---------------------------------------------------------------------------
# Schema validation — refusal paths
# ---------------------------------------------------------------------------


def test_base_manifest_is_valid():
    assert_valid(base_manifest())


def test_schema_version_refusal():
    m = base_manifest()
    m["schema_version"] = "e5-cell-manifest/2"
    violations = e5.validate_cell_manifest(m)
    assert len(violations) == 1  # fail closed: nothing else is interpreted
    assert_violation(violations, "schema_version")


def test_protocol_id_refusal():
    m = base_manifest()
    m["protocol_id"] = "P-BENCH-2"
    assert_violation(e5.validate_cell_manifest(m), "protocol_id")


def test_era_stamp_refusal():
    m = base_manifest()
    m["era"]["cpu_kernel"] = "E5-cpu-kernel"
    assert_violation(e5.validate_cell_manifest(m), "era.cpu_kernel")


def test_port_range_refusal():
    m = base_manifest()
    m["instances"][0]["port"] = 18070  # eval lane port — must be refused
    assert_violation(e5.validate_cell_manifest(m), "outside the E5 bench range")
    m["instances"][0]["port"] = 8080  # prod frontdoor quarter — refused
    assert_violation(e5.validate_cell_manifest(m), "outside the E5 bench range")
    m["instances"][0]["port"] = 20000  # above the bench range — refused
    assert_violation(e5.validate_cell_manifest(m), "outside the E5 bench range")


def test_port_collision_refusal():
    m = base_manifest()
    m["instances"][1]["port"] = m["instances"][0]["port"]
    assert_violation(e5.validate_cell_manifest(m), "collides")


def test_overlapping_cpuset_refusal():
    m = base_manifest()
    m["instances"][1]["cpu_list"] = m["instances"][0]["cpu_list"]
    assert_violation(e5.validate_cell_manifest(m), "overlaps")


def test_half_overlapping_quarter_refusal():
    m = base_manifest()
    # half0 superset overlaps the Q0A quarter in instances[0]
    m["instances"][1]["cpu_list"] = e5.CPUSET_HALF0
    m["instances"][1]["threads"] = 96
    assert_violation(e5.validate_cell_manifest(m), "overlaps")


def test_threads_cpuset_mismatch_refusal():
    m = base_manifest()
    m["instances"][0]["threads"] = 96  # Q0A is a 48-CPU set
    assert_violation(e5.validate_cell_manifest(m), "cpuset cardinality")


def test_bad_cpulist_syntax_refusal():
    m = base_manifest()
    m["instances"][0]["cpu_list"] = "0-23,abc"
    assert_violation(e5.validate_cell_manifest(m), "cpulist")


def test_np_ladder_enforcement():
    m = by_id("qwen36_q8_0-C1-np8")
    m["np"] = 3
    m["cell_id"] = "qwen36_q8_0-C1-np3"
    m["ctx"] = e5.compute_ctx(3)
    assert_violation(e5.validate_cell_manifest(m), "pre-registered C1 ladder")


def test_np_total_in_flight_cap():
    m = base_manifest()  # C3, 4 instances
    m["np"] = 16  # 4*16 = 64 > 43 (and 16 is off the C3 ladder)
    m["cell_id"] = "qwen36_q8_0-C3-np16"
    m["ctx"] = e5.compute_ctx(16)
    violations = e5.validate_cell_manifest(m)
    assert_violation(violations, "pre-registered C3 ladder")
    assert_violation(violations, "exceeds the cap 43")


def test_instance_count_per_config():
    m = base_manifest()  # C3 requires 4 instances
    m["instances"] = m["instances"][:3]
    assert_violation(e5.validate_cell_manifest(m), "requires exactly 4 instance(s)")


def test_ctx_sizing_arithmetic():
    assert e5.compute_ctx(1) == 8192
    assert e5.compute_ctx(2) == 8192
    assert e5.compute_ctx(4) == 8192
    assert e5.compute_ctx(8) == 16384
    assert e5.compute_ctx(16) == 32768
    assert e5.compute_ctx(32) == 65536


def test_ctx_mismatch_refusal():
    m = base_manifest()  # np=8 → ctx must be 16384
    m["ctx"] = 32768
    assert_violation(e5.validate_cell_manifest(m), "ctx")
    m["ctx"] = 8192
    assert_violation(e5.validate_cell_manifest(m), "16384")


def test_per_stream_ctx_constant_refusal():
    m = base_manifest()
    m["per_stream_ctx"] = 1024
    assert_violation(e5.validate_cell_manifest(m), "per_stream_ctx")


def test_interleave_refused_off_full_shape():
    m = base_manifest()
    m["instances"][0]["numactl_policy"] = "interleave=all"  # a 48t quarter
    assert_violation(e5.validate_cell_manifest(m), "ONLY legal on the full-machine shape")


def test_full_shape_requires_interleave():
    m = by_id("qwen36_27b_q8-C1-np1-scout-full")
    m["instances"][0]["numactl_policy"] = "none"
    assert_violation(e5.validate_cell_manifest(m), "requires numactl_policy 'interleave=all'")


def test_ssm_kv_unified_refusal():
    m = by_id("qwen3_next_80b-C1-np8")
    m["kv"]["kv_unified"] = True
    assert_violation(e5.validate_cell_manifest(m), "REFUSED for SSM/hybrid")


def test_w0_decision_grade_refusal():
    m = by_id("qwen36_q8_0-C1-np16-scout")
    m["decision_grade_intent"] = True
    violations = e5.validate_cell_manifest(m)
    assert_violation(violations, "W0 scout cells MUST set decision_grade_intent=false")


def test_scout_cap_never_decision_grade():
    m = base_manifest()
    m["prompt_caps"]["n_predict"] = 64  # scout cap on a decision-grade cell
    assert_violation(e5.validate_cell_manifest(m), "decision_grade_intent=true requires")


def test_n_predict_whitelist():
    m = base_manifest()
    m["prompt_caps"]["n_predict"] = 128
    assert_violation(e5.validate_cell_manifest(m), "n_predict")


def test_max_prompt_chars_guard():
    m = base_manifest()
    m["prompt_caps"]["max_prompt_chars"] = 200000  # would admit the 101k-char prompt
    assert_violation(e5.validate_cell_manifest(m), "max_prompt_chars")


def test_pinned_qids_exact():
    m = base_manifest()
    m["prompt_batch"]["qids"] = m["prompt_batch"]["qids"][:-1]
    assert_violation(e5.validate_cell_manifest(m), "43-qid E1 pinned batch")
    m = base_manifest()
    m["prompt_batch"]["qids"][0] = "tulving_Udefault_Sdefault_seed0_ch-001_q0270"
    assert_violation(e5.validate_cell_manifest(m), "43-qid E1 pinned batch")


def test_selection_must_be_pinned_qids():
    m = base_manifest()
    m["prompt_batch"]["selection"] = "tier_seed"
    assert_violation(e5.validate_cell_manifest(m), "re-sampling tier/seed")


def test_pinned_qid_constant_shape():
    qids = e5.E1_PINNED_QIDS
    assert len(qids) == 43
    assert len(set(qids)) == 43
    assert qids[0] == "debugbench_fizz-buzz_python"
    assert qids[-1] == "simpleqa_general_02228"


def test_pinned_qids_match_e1_artifact_when_present():
    artifact = _RESEARCH_ROOT / e5.E1_PINNED_FROM
    if not artifact.is_file():
        return  # artifact not on this checkout — constant shape covered above
    got = [json.loads(line)["qid"] for line in artifact.read_text().splitlines() if line.strip()]
    assert got == e5.E1_PINNED_QIDS


def test_spec_dec_refusal_paths():
    m = base_manifest()
    m["spec_dec"]["disabled_reason"] = "leftover"
    assert_violation(e5.validate_cell_manifest(m), "disabled_reason must be null")

    m = base_manifest()
    m["spec_dec"]["enabled"] = False
    m["spec_dec"]["disabled_reason"] = None
    assert_violation(e5.validate_cell_manifest(m), "never silent")

    m = base_manifest()
    m["spec_dec"]["device_draft"] = "ROCm0"
    assert_violation(e5.validate_cell_manifest(m), "device_draft")

    m = base_manifest()
    m["spec_dec"]["record_accept_rate"] = False
    assert_violation(e5.validate_cell_manifest(m), "record_accept_rate")


def test_env_expectation_refusal():
    m = base_manifest()
    m["env_expectation"]["ggml_iqk"] = "0"
    assert_violation(e5.validate_cell_manifest(m), "ggml_iqk")

    m = base_manifest()
    m["env_expectation"]["kmp_blocktime"] = "200"
    assert_violation(e5.validate_cell_manifest(m), "kmp_blocktime")

    m = base_manifest()
    m["env_expectation"]["omp_source"] = "private DEFAULT_ENV"
    assert_violation(e5.validate_cell_manifest(m), "canonical_recipe")


def test_e1_parity_flags_required():
    m = base_manifest()
    m["mlock"] = False
    assert_violation(e5.validate_cell_manifest(m), "mlock")
    m = base_manifest()
    m["jinja"] = False
    assert_violation(e5.validate_cell_manifest(m), "jinja")


def test_kv_flags_required():
    m = base_manifest()
    m["kv"]["type_v"] = "f16"
    assert_violation(e5.validate_cell_manifest(m), "q8_0")
    m = base_manifest()
    m["kv"]["flash_attn"] = False
    assert_violation(e5.validate_cell_manifest(m), "flash_attn")


def test_cell_id_prefix_enforced():
    m = base_manifest()
    m["cell_id"] = "qwen36_q8_0-C3-np16"  # wrong np in the id
    assert_violation(e5.validate_cell_manifest(m), "cell_id")


def test_parse_cpulist():
    assert len(e5.parse_cpulist("0-23,96-119")) == 48
    assert len(e5.parse_cpulist("0-47,96-143")) == 96
    assert len(e5.parse_cpulist("0-95")) == 96
    assert e5.parse_cpulist("0-23,abc") is None
    assert e5.parse_cpulist("5-2") is None
    assert e5.parse_cpulist("") is None


# ---------------------------------------------------------------------------
# Grid generation — completeness vs the pre-registered spec table
# ---------------------------------------------------------------------------


def test_grid_all_cells_validate_clean():
    for cell in grid():
        violations = e5.validate_cell_manifest(cell)
        assert violations == [], f"{cell['cell_id']}: {violations}"


def test_grid_unique_cell_ids():
    ids = [c["cell_id"] for c in grid()]
    assert len(ids) == len(set(ids))


def test_grid_total_counts():
    cells = grid()
    # 116 base + 5 E1-parity twins (operator sampling decision 2026-07-23):
    # qwen36 C1@1/C3@1, dense C1@1/C3@1 + scout-full@1.
    assert len(cells) == 121
    scout = [c for c in cells if c["window"] == "W0"]
    stage_b = [c for c in cells if c["window"] != "W0"]
    assert len(scout) == 69
    assert len(stage_b) == 52


def test_grid_counts_per_model_and_config():
    counts: dict[tuple[str, str, str], int] = {}
    for c in grid():
        key = (c["model_key"], c["config_id"], "W0" if c["window"] == "W0" else "B")
        counts[key] = counts.get(key, 0) + 1
    expected = {
        # qwen36_q8_0: scout ladders + kvu probe; Stage-B W1 set
        ("qwen36_q8_0", "C1", "W0"): 7,  # 6 ladder + 1 kvu probe
        ("qwen36_q8_0", "C1b", "W0"): 5,
        ("qwen36_q8_0", "C2", "W0"): 5,
        ("qwen36_q8_0", "C3", "W0"): 4,
        ("qwen36_q8_0", "C1", "B"): 6,  # +e1parity twin @1
        ("qwen36_q8_0", "C1b", "B"): 3,
        ("qwen36_q8_0", "C2", "B"): 2,
        ("qwen36_q8_0", "C3", "B"): 5,  # +e1parity twin @1
        # dense control: + the 2-cell full-machine scout pair
        ("qwen36_27b_q8", "C1", "W0"): 9,  # 6 ladder + 2 scout-full pair + scout-full e1parity twin
        ("qwen36_27b_q8", "C1b", "W0"): 5,
        ("qwen36_27b_q8", "C2", "W0"): 5,
        ("qwen36_27b_q8", "C3", "W0"): 4,
        ("qwen36_27b_q8", "C1", "B"): 6,  # +e1parity twin @1
        ("qwen36_27b_q8", "C1b", "B"): 3,
        ("qwen36_27b_q8", "C2", "B"): 2,
        ("qwen36_27b_q8", "C3", "B"): 5,  # +e1parity twin @1
        # ingest arm: no C2 anywhere
        ("qwen3_next_80b", "C1", "W0"): 6,
        ("qwen3_next_80b", "C1b", "W0"): 5,
        ("qwen3_next_80b", "C3", "W0"): 4,
        ("qwen3_next_80b", "C1", "B"): 5,
        ("qwen3_next_80b", "C1b", "B"): 3,
        ("qwen3_next_80b", "C3", "B"): 4,
        # gemma: full + quarters only
        ("gemma4_26b_a4b_q4km_mtp", "C1", "W0"): 6,
        ("gemma4_26b_a4b_q4km_mtp", "C3", "W0"): 4,
        ("gemma4_26b_a4b_q4km_mtp", "C1", "B"): 4,
        ("gemma4_26b_a4b_q4km_mtp", "C3", "B"): 4,
    }
    assert counts == expected


def test_grid_stage_b_window_mapping():
    windows = {c["model_key"]: set() for c in grid()}
    for c in grid():
        if c["window"] != "W0":
            windows[c["model_key"]].add(c["window"])
    assert windows["qwen36_q8_0"] == {"W1"}
    assert windows["gemma4_26b_a4b_q4km_mtp"] == {"W2"}
    assert windows["qwen36_27b_q8"] == {"W3"}
    assert windows["qwen3_next_80b"] == {"W4"}


def test_grid_scout_flags():
    for c in grid():
        is_twin = "e1_parity_anchor" in (c.get("stage_b_families") or [])
        if c["window"] == "W0":
            assert c["decision_grade_intent"] is False, c["cell_id"]
            assert c["prompt_caps"]["n_predict"] == 64, c["cell_id"]
        elif is_twin:
            # E1-parity twins live in the stage-B windows for scheduling but
            # are continuity reads, never decision cells.
            assert c["decision_grade_intent"] is False, c["cell_id"]
        else:
            assert c["decision_grade_intent"] is True, c["cell_id"]
            assert c["prompt_caps"]["n_predict"] == 256, c["cell_id"]


def test_grid_in_flight_cap_holds_everywhere():
    for c in grid():
        assert len(c["instances"]) * c["np"] <= 43, c["cell_id"]


def test_grid_ctx_rule_holds_everywhere():
    for c in grid():
        assert c["per_stream_ctx"] == 2048, c["cell_id"]
        assert c["ctx"] == max(8192, 2048 * c["np"]), c["cell_id"]


def test_grid_ports_in_range_no_collisions_deterministic():
    for c in grid():
        ports = [inst["port"] for inst in c["instances"]]
        assert len(ports) == len(set(ports)), c["cell_id"]
        base = e5.MODELS[c["model_key"]]["port_base"]
        for idx, inst in enumerate(c["instances"]):
            assert 19000 <= inst["port"] <= 19999, c["cell_id"]
            assert inst["port"] == base + idx, c["cell_id"]


def test_grid_interleave_only_on_full():
    for c in grid():
        for inst in c["instances"]:
            if inst["cpu_list"] == "0-95":
                assert inst["numactl_policy"] == "interleave=all", c["cell_id"]
            else:
                assert inst["numactl_policy"] == "none", c["cell_id"]


def test_gemma_no_half_invariant():
    quarters = {e5.CPUSET_Q0A, e5.CPUSET_Q0B, e5.CPUSET_Q1A, e5.CPUSET_Q1B}
    gemma_cells = [c for c in grid() if c["model_key"] == "gemma4_26b_a4b_q4km_mtp"]
    assert gemma_cells
    for c in gemma_cells:
        assert c["config_id"] in ("C1", "C3"), c["cell_id"]  # no C1b/C2 shapes
        for inst in c["instances"]:
            if inst["cpu_list"] == "0-95":
                assert inst["numactl_policy"] == "interleave=all", c["cell_id"]
                assert inst["threads"] == 96, c["cell_id"]
            else:
                assert inst["cpu_list"] in quarters, (
                    f"{c['cell_id']}: gemma half shape {inst['cpu_list']!r} is forbidden"
                )
        assert c["ubatch_size"] == 512, c["cell_id"]  # MTP recipe override


def test_gemma_spec_recipe_fields():
    for c in grid():
        if c["model_key"] != "gemma4_26b_a4b_q4km_mtp":
            continue
        spec = c["spec_dec"]
        assert spec["enabled"] is True
        assert spec["spec_type"] == "draft-mtp"
        assert spec["draft_max"] == 2
        assert spec["draft_p_min"] == 0.0
        assert spec["draft_p_split"] == 0
        assert spec["threads_draft"] == 16
        assert spec["draft_model_path"] == (
            "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf"
        )
        assert spec["device_draft"] == "none"
        assert spec["record_accept_rate"] is True


def test_qwen36_spec_recipe_fields():
    for c in grid():
        if c["model_key"] not in ("qwen36_q8_0", "qwen36_27b_q8"):
            continue
        spec = c["spec_dec"]
        assert spec["enabled"] is True, c["cell_id"]
        assert spec["spec_type"] == "draft-mtp", c["cell_id"]
        assert spec["draft_max"] == 4, c["cell_id"]
        assert spec["draft_p_split"] == 0, c["cell_id"]
        assert spec["draft_model_path"] is None, c["cell_id"]  # NEXTN self-draft: omit -md
        assert spec["device_draft"] == "none", c["cell_id"]


def test_qwen3_next_spec_disabled_and_kv_off():
    cells = [c for c in grid() if c["model_key"] == "qwen3_next_80b"]
    assert cells
    for c in cells:
        assert c["architecture"] == "ssm_moe_hybrid", c["cell_id"]
        assert c["spec_dec"]["enabled"] is False, c["cell_id"]
        assert c["spec_dec"]["disabled_reason"], c["cell_id"]  # never silent
        assert c["kv"]["kv_unified"] is False, c["cell_id"]


def test_grid_kvu_probe():
    kvu_cells = [c for c in grid() if c["kv"]["kv_unified"]]
    assert len(kvu_cells) == 1  # split KV is the default everywhere else
    probe = kvu_cells[0]
    assert probe["cell_id"] == "qwen36_q8_0-C1-np16-scout-kvu"
    assert probe["window"] == "W0"
    assert probe["config_id"] == "C1"
    assert probe["np"] == 16
    assert probe["decision_grade_intent"] is False
    # its split-KV pair cell exists
    by_id("qwen36_q8_0-C1-np16-scout")


def test_grid_dense_scout_shape_pair():
    for np in (1, 8):
        full = by_id(f"qwen36_27b_q8-C1-np{np}-scout-full")
        assert full["window"] == "W0"
        assert full["decision_grade_intent"] is False
        assert len(full["instances"]) == 1
        assert full["instances"][0]["cpu_list"] == "0-95"
        assert full["instances"][0]["numactl_policy"] == "interleave=all"
        # the half0 side of the pair exists in the scout ladder
        half = by_id(f"qwen36_27b_q8-C1-np{np}-scout")
        assert half["instances"][0]["cpu_list"] == e5.CPUSET_HALF0


def test_grid_ubatch_per_shape():
    for c in grid():
        if c["model_key"] == "gemma4_26b_a4b_q4km_mtp":
            assert c["ubatch_size"] == 512, c["cell_id"]
        elif all(inst["threads"] == 96 for inst in c["instances"]):
            assert c["ubatch_size"] == 8192, c["cell_id"]
        else:
            assert c["ubatch_size"] == 512, c["cell_id"]


def test_grid_config_shapes():
    for c in grid():
        shapes = [inst["cpu_list"] for inst in c["instances"]]
        if c["config_id"] == "C1b":
            assert shapes == [e5.CPUSET_HALF0, e5.CPUSET_HALF1], c["cell_id"]
        elif c["config_id"] == "C2":
            assert shapes == [e5.CPUSET_Q1A, e5.CPUSET_Q1B], c["cell_id"]
        elif c["config_id"] == "C3":
            assert shapes == [
                e5.CPUSET_Q0A, e5.CPUSET_Q0B, e5.CPUSET_Q1A, e5.CPUSET_Q1B,
            ], c["cell_id"]


def test_grid_stage_b_iso_t_families():
    # Whole-machine provisioning pairs {C1b@T/2 vs C3@T/4}, T in {8,16,32}
    for t in (8, 16, 32):
        c1b = by_id(f"qwen36_q8_0-C1b-np{t // 2}")
        c3 = by_id(f"qwen36_q8_0-C3-np{t // 4}")
        assert f"whole_machine_T{t}" in c1b["stage_b_families"]
        assert f"whole_machine_T{t}" in c3["stage_b_families"]
    # Half-machine mechanism pairs {C1@T vs C2@T/2}, T in {16,32}
    for t in (16, 32):
        c1 = by_id(f"qwen36_q8_0-C1-np{t}")
        c2 = by_id(f"qwen36_q8_0-C2-np{t // 2}")
        assert f"mechanism_T{t}" in c1["stage_b_families"]
        assert f"mechanism_T{t}" in c2["stage_b_families"]
    # Scaling pairs {C1@K vs C1b@K}, K in {4,8}
    for k in (4, 8):
        assert f"scaling_K{k}" in by_id(f"qwen36_q8_0-C1-np{k}")["stage_b_families"]
        assert f"scaling_K{k}" in by_id(f"qwen36_q8_0-C1b-np{k}")["stage_b_families"]
    # Anchors
    assert "anchor_C1" in by_id("qwen36_q8_0-C1-np1")["stage_b_families"]
    assert "anchor_C3" in by_id("qwen36_q8_0-C3-np1")["stage_b_families"]
    # Gemma whole-machine pairs are C1full@T vs C3@T/4
    for t in (8, 16, 32):
        gfull = by_id(f"gemma4_26b_a4b_q4km_mtp-C1-np{t}")
        gq = by_id(f"gemma4_26b_a4b_q4km_mtp-C3-np{t // 4}")
        assert f"whole_machine_T{t}" in gfull["stage_b_families"]
        assert f"whole_machine_T{t}" in gq["stage_b_families"]


# ---------------------------------------------------------------------------
# CLI round-trips (tempdir only; no filesystem side effects outside tmp)
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = e5.main(argv)
    return rc, buf.getvalue()


def test_cli_generate_writes_grid_grouped_by_model():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out = _run_cli(["generate", "--output-dir", tmp, "--skip-file-checks"])
        assert rc == 0, out
        root = Path(tmp)
        model_dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
        assert model_dirs == sorted(e5.MODELS)
        files = list(root.glob("*/*.json"))
        assert len(files) == 121
        # spot-check one file round-trips through the validator
        sample = json.loads((root / "qwen36_q8_0" / "qwen36_q8_0-C3-np8.json").read_text())
        assert e5.validate_cell_manifest(sample) == []
        assert "wrote 121 cell manifests" in out


def test_cli_validate_exit_codes():
    with tempfile.TemporaryDirectory() as tmp:
        good_path = Path(tmp) / "good.json"
        good_path.write_text(json.dumps(base_manifest()))
        rc, out = _run_cli(["validate", str(good_path)])
        assert rc == 0
        assert "OK" in out

        bad = base_manifest()
        bad["instances"][0]["port"] = 8080  # prod port → refusal
        bad_path = Path(tmp) / "bad.json"
        bad_path.write_text(json.dumps(bad))
        rc, out = _run_cli(["validate", str(good_path), str(bad_path)])
        assert rc == 1
        assert "outside the E5 bench range" in out

        junk_path = Path(tmp) / "junk.json"
        junk_path.write_text("{not json")
        rc, out = _run_cli(["validate", str(junk_path)])
        assert rc == 1
        assert "invalid JSON" in out


# ---------------------------------------------------------------------------
# Stdlib runner (used when pytest is not installed)
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = 0
    failures: list[str] = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0




def test_sampling_regime_operator_decision():
    """Operator decision 2026-07-23: production temp+seed42 everywhere;
    temperature-0 only in the five -e1parity twins of the E1-tied anchors;
    ingest production is already greedy (seed pinned)."""
    cells = grid()
    twins = [c for c in cells if c["cell_id"].endswith("-e1parity")]
    assert sorted(c["cell_id"] for c in twins) == [
        "qwen36_27b_q8-C1-np1-e1parity",
        "qwen36_27b_q8-C1-np1-scout-full-e1parity",
        "qwen36_27b_q8-C3-np1-e1parity",
        "qwen36_q8_0-C1-np1-e1parity",
        "qwen36_q8_0-C3-np1-e1parity",
    ]
    twin_ids = {c["cell_id"] for c in twins}
    for c in cells:
        s = c.get("sampling")
        assert isinstance(s, dict), c["cell_id"]
        assert s.get("seed") == 42, c["cell_id"]
        if c["cell_id"] in twin_ids:
            assert s["regime"] == "e1_parity_greedy", c["cell_id"]
            assert s["temperature"] == 0.0, c["cell_id"]
            assert "e1_parity_anchor" in c["stage_b_families"], c["cell_id"]
            assert c["decision_grade_intent"] is False, c["cell_id"]
        else:
            assert s["regime"] == "production", c["cell_id"]
            if c["model_key"] == "qwen3_next_80b":
                assert s["temperature"] == 0.0, c["cell_id"]  # production greedy
            else:
                assert s["temperature"] == 0.3, c["cell_id"]


def test_e1_parity_twins_only_for_e1_tied_models():
    for c in grid():
        if c["cell_id"].endswith("-e1parity"):
            assert c["model_key"] in ("qwen36_q8_0", "qwen36_27b_q8"), c["cell_id"]




def test_request_endpoint_production_chat_with_per_model_template_kwargs():
    """2026-07-23 think-truncation incident: main cells run the production
    chat+template recipe; qwen3x carries enable_thinking=false, the 80B GGUF
    template ignores the kwarg (registry: pass nothing), gemma has no toggle;
    -e1parity twins keep raw /completion (the E1 shape)."""
    for c in grid():
        is_twin = c["cell_id"].endswith("-e1parity")
        if is_twin:
            assert c["request_endpoint"] == "completion", c["cell_id"]
            assert c["chat_template_kwargs"] == {}, c["cell_id"]
            continue
        assert c["request_endpoint"] == "chat_completions", c["cell_id"]
        if c["model_key"] in ("qwen36_q8_0", "qwen36_27b_q8"):
            assert c["chat_template_kwargs"] == {"enable_thinking": False}, c["cell_id"]
        else:
            assert c["chat_template_kwargs"] == {}, c["cell_id"]


# ---------------------------------------------------------------------------
# Era derivation (A7 Token 2 Block B, 2026-08-11). The constants these replace
# were correct when written and silently wrong two days later.
# ---------------------------------------------------------------------------


def test_no_era_id_is_hardcoded_in_the_generator():
    """The defect was the PRESENCE of a constant, not its value.

    Any era id written here is correct only until the next cutover — E6 lasted
    five days. Prose mentions in the explanatory comment are fine; string
    literals that code could read are not.
    """
    # Scan CODE, not prose: the module comment quotes the two constants this
    # replaced, and keeping that history is the point. Strip whole-line comments
    # and check what is left.
    code = "\n".join(
        line for line in Path(e5.__file__).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    for literal in ("E6-cpu-kernel", "E7-eval-instrument", "E8-cpu-kernel", "E9-cpu-kernel"):
        assert f'"{literal}"' not in code and f"'{literal}'" not in code, (
            f"{literal} is a string literal in the generator; the era must be derived"
        )


def test_kernel_era_is_binary_witnessed_not_scope_latest():
    """Scope alone gives the wrong kernel for the entire W1/W2/W4 window.

    Since the consolidated era token of 2026-08-11T21:35Z, `cpu_bench` carries
    both kernel cutovers and E8-cpu-bench-throttle-scope, an ELIGIBILITY
    boundary. A latest-in-scope lookup returns the eligibility row for any
    instant in 2026-07-29..2026-08-10 — precisely the window of the six known
    mis-stamped run manifests, so it would hand them a second wrong answer.
    """
    import instrument_era as ie

    eras = ie.load_registry()
    w1_instant = "2026-07-29T15:47:29Z"
    assert ie.derive_era("cpu_bench", w1_instant, eras) == "E8-cpu-bench-throttle-scope"
    assert ie.derive_kernel_era(w1_instant, eras) == "E8-cpu-kernel"
    # The instants either side are unaffected, so the discriminator is not just
    # shifting the error somewhere else.
    assert ie.derive_kernel_era("2026-07-23T19:03:41Z", eras) == "E6-cpu-kernel"
    assert ie.derive_kernel_era("2026-08-11T22:00:00Z", eras) == "E9-cpu-kernel"


def test_kernel_derivation_fails_closed_in_the_unwitnessed_window():
    """[2026-06-26T22:07:11Z, 2026-07-20T13:30:13Z) must REFUSE, loudly.

    E5-cpu-kernel was deliberately excluded from
    RATIFY-CPU-BENCH-BINARY-VERSION-20260811: its registry note records no
    binary version and no commit sha, and inventing one is the exact failure the
    repair exists to stop. So there is a real window with no witnessed kernel.
    The requirement is that it raises rather than silently picking the nearest
    neighbour — a silent neighbour is indistinguishable from a correct answer,
    which is how the original mis-stamp survived a cutover.

    No banked manifest falls in this window (E5 pre-registration begins
    2026-07-23), so this test is the only thing standing between the gap and a
    future silent fallback.
    """
    import instrument_era as ie

    eras = ie.load_registry()
    for instant in (
        "2026-06-26T22:07:11Z",  # exactly E5-cpu-kernel's own boundary
        "2026-07-01T00:00:00Z",  # mid-window
        "2026-07-20T13:30:12Z",  # one second before E6 opens
    ):
        try:
            resolved = ie.derive_kernel_era(instant, eras)
        except ie.EraDerivationError as exc:
            text = str(exc)
            assert "binary_version" in text, "the refusal must say WHY"
            assert "REFUSING" in text
        else:
            raise AssertionError(
                f"derive_kernel_era({instant}) silently returned {resolved!r} — it must "
                "refuse; a neighbouring era here is a stamp naming an instrument that "
                "nothing witnessed"
            )
    # And the boundary itself opens exactly on time, so the refusal window is not
    # one second too wide.
    assert ie.derive_kernel_era("2026-07-20T13:30:13Z", eras) == "E6-cpu-kernel"


def test_era_for_binary_binds_to_the_witness_and_refuses_an_unknown_one():
    """binary_version is the only field that witnesses what actually executed."""
    import instrument_era as ie

    eras = ie.load_registry()
    assert ie.era_for_binary(10098, eras) == "E6-cpu-kernel"
    assert ie.era_for_binary(10107, eras) == "E8-cpu-kernel"
    # The attestation records a multi-line blob, not a bare int.
    assert ie.era_for_binary("version: 10125 (0db32c06e)\nbuilt with GNU", eras) == "E9-cpu-kernel"
    try:
        ie.era_for_binary(99999, eras)
    except ie.EraDerivationError as exc:
        assert "REFUSING" in str(exc)
    else:
        raise AssertionError("an unregistered binary must not resolve to any era")


def test_generated_manifests_carry_generated_at_and_the_derived_era():
    cell = e5.make_cell(window="W1", model_key="qwen36_q8_0", config_id="C1", np=1,
                       decision_grade_intent=True, n_predict=e5.N_PREDICT_STAGE_B)
    assert cell["generated_at"], "a manifest must witness its own pre-registration instant"
    import instrument_era as ie

    eras = ie.load_registry()
    assert cell["era"]["cpu_kernel"] == ie.derive_kernel_era(cell["generated_at"], eras)
    assert cell["era"]["eval_instrument"] == ie.derive_era(
        "eval_quality", cell["generated_at"], eras
    )
    assert e5.validate_cell_manifest(cell) == []


def test_legacy_manifests_without_generated_at_still_validate():
    """The 191 pre-registered files must stay valid — this was campaign-breaking.

    `revalidate_cells` is fail-closed and runs the validator over every manifest
    it loads, so an equality-to-current-era rule makes the sweep refuse to start
    against the whole corpus, including the un-executed templates W2/W4 depend on.
    """
    cell = e5.make_cell(window="W1", model_key="qwen36_q8_0", config_id="C1", np=1,
                       decision_grade_intent=True, n_predict=e5.N_PREDICT_STAGE_B)
    legacy = copy.deepcopy(cell)
    legacy.pop("generated_at", None)
    legacy["era"] = {
        "cpu_kernel": "E6-cpu-kernel",
        "eval_instrument": "E7-eval-instrument",
        "source": legacy["era"]["source"],
    }
    assert e5.validate_cell_manifest(legacy) == []


def test_a_dated_manifest_with_the_wrong_era_is_rejected():
    """The loose legacy path must not become a blanket amnesty.

    Once a manifest says when it was registered, exactly one era is correct for
    it — otherwise this repair would be the 'permanently whitelist the stale
    pair' shape that got the earlier package refused.
    """
    cell = e5.make_cell(window="W1", model_key="qwen36_q8_0", config_id="C1", np=1,
                       decision_grade_intent=True, n_predict=e5.N_PREDICT_STAGE_B)
    cell["generated_at"] = "2026-08-11T22:00:00Z"  # v9 era
    cell["era"] = dict(cell["era"], cpu_kernel="E6-cpu-kernel")  # claims v7
    problems = e5.validate_cell_manifest(cell)
    assert any("cpu_kernel" in p for p in problems), problems


def test_an_invented_era_id_is_rejected_even_without_generated_at():
    cell = e5.make_cell(window="W1", model_key="qwen36_q8_0", config_id="C1", np=1,
                       decision_grade_intent=True, n_predict=e5.N_PREDICT_STAGE_B)
    cell.pop("generated_at", None)
    cell["era"] = dict(cell["era"], cpu_kernel="E42-invented-kernel")
    problems = e5.validate_cell_manifest(cell)
    assert any("cpu_kernel" in p for p in problems), problems


if __name__ == "__main__":
    raise SystemExit(_run_all())
