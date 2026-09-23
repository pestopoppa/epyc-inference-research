"""Out-of-band instrumented sibling profiling; real fixture dumps, no build or launch.

The three fixtures under `fixtures/ds41-*.json` are the ACTUAL dumps from the
2026-09-23 DeepSeek-V4.1 smoke on the instrumented sibling build, not hand-written
approximations of the writers. The only edit is that `ds41-ggml-cpu-prof.json` keeps
the six heaviest of its 3772 `nodes[]` rows so the fixture stays committable; every key
this parser reads (`ops`, `paths`, `shapes`, the top level) is untouched.

That smoke also carries two states worth keeping: `host.us_decode_step_span == 0` (the
engram gather share is genuinely underivable from it) and `n_eval == 1` under
speculation. A reader that only works on a clean run is a reader that breaks in the
field.
"""
import json
from pathlib import Path

import pytest

from ..controller import build_recipe as br
from . import node_profile as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NODE_RAW = (FIXTURES / "ds41-ggml-cpu-prof.json").read_bytes()
HOST_RAW = (FIXTURES / "ds41-llama-host-prof.json").read_bytes()
ENGRAM_RAW = (FIXTURES / "ds41-engram-prof.json").read_bytes()
NODE_DUMP = json.loads(NODE_RAW)
HOST_DUMP = json.loads(HOST_RAW)
ENGRAM_DUMP = json.loads(ENGRAM_RAW)
TOTAL_WALL_US = 42529.311  # ds41-ggml-cpu-prof.json total_wall_us, per graph eval


def write_dumps(directory, *, node=NODE_RAW, host=HOST_RAW, engram=ENGRAM_RAW):
    directory.mkdir(parents=True, exist_ok=True)
    paths = np.dump_paths(directory)
    for name, body in (("node", node), ("host", host), ("engram", engram)):
        if body is None:
            continue
        Path(paths[name]).write_bytes(body if isinstance(body, bytes) else body.encode())
    return paths


class _Template:
    cpu_list = None


class _Arm:
    """The only surface of a resolved recipe `profile_loop` touches."""

    def __init__(self, env, build_dir):
        self.launch_env = tuple(sorted(env.items()))
        self.build_dir = str(build_dir)
        self.port = 18080
        self.template = _Template()


class _Prompt:
    prompt_id = "p0"


class _Prompts:
    prompts = (_Prompt(),)

    def requests(self, ids, template):
        assert ids == ("p0",)
        return (("p0", b'{"prompt":"x"}'),)


def test_profiling_recipe_is_the_base_plus_the_compile_gate():
    base = dict(br.NATIVE_CPU_RECIPE.cmake_defines())
    variant = dict(br.NATIVE_CPU_NODE_PROFILE_RECIPE.cmake_defines())
    assert variant.pop("GGML_CPU_PROF") == "ON"
    assert variant == base
    assert br.NATIVE_CPU_NODE_PROFILE_RECIPE.sha256() != br.NATIVE_CPU_RECIPE.sha256()
    assert br.recipe_for(br.NATIVE_CPU_NODE_PROFILE_RECIPE.name) \
        is br.NATIVE_CPU_NODE_PROFILE_RECIPE
    # The measured recipe must not have grown the define along the way.
    assert "GGML_CPU_PROF" not in base


def test_profiling_build_dir_is_a_sibling_not_a_child(tmp_path):
    assert np.profiling_build_dir(tmp_path / "build-cpu") == tmp_path / "build-cpu-prof"


def test_launch_env_carries_the_gate_and_all_three_dump_paths(tmp_path):
    env = np.launch_env(tmp_path, level=2)
    paths = np.dump_paths(tmp_path)
    assert env == {np.CPU_PROF_GATE_ENV: "1",
                   np.CPU_PROF_JSON_ENV: paths["node"],
                   np.HOST_PROF_JSON_ENV: paths["host"],
                   np.ENGRAM_JSON_ENV: paths["engram"],
                   np.ENGRAM_LEVEL_ENV: "2"}
    # The writer's legacy path alias is never SET -- one path, one spelling -- but it is
    # still refused on a measured arm.
    assert np.ENGRAM_LEGACY_PATH_ENV not in env
    assert np.ENGRAM_LEGACY_PATH_ENV in np.PROFILE_ENV_KEYS
    assert set(env) < set(np.PROFILE_ENV_KEYS)
    with pytest.raises(np.NodeProfileRefused):
        np.launch_env(tmp_path, level=3)


def test_measured_launch_env_is_refused_when_it_carries_the_instrument(tmp_path):
    np.refuse_instrumented_measurement(_Arm({"LD_LIBRARY_PATH": "/x/bin"}, tmp_path))
    for key in (np.CPU_PROF_GATE_ENV, np.ENGRAM_JSON_ENV, np.ENGRAM_LEGACY_PATH_ENV):
        dirty = _Arm({"LD_LIBRARY_PATH": "/x/bin", key: "1"}, tmp_path)
        with pytest.raises(np.NodeProfileRefused, match="instrumentation environment"):
            np.refuse_instrumented_measurement(dirty)


def test_node_dump_op_shares_match_the_writers_own_pct_wall():
    node = np.parse_node_dump(NODE_RAW)
    assert node["total_wall_us"] == TOTAL_WALL_US
    assert node["n_threads"] == 48 and node["graph_evals_accumulated"] == 132
    assert node["ops"][0]["op"] == "MUL_MAT_ID"  # ranked by wall, not by dump order
    for row in node["ops"]:
        # `pct_wall` is the writer's own 0-100 percentage of the same denominator; our
        # fraction must reproduce it, which is what makes the share readable at all.
        assert row["wall_fraction"] * 100 == pytest.approx(
            next(r["pct_wall"] for r in NODE_DUMP["ops"] if r["op"] == row["op"]),
            abs=5e-5)
    # Not 1.0: fused RMS_NORM+MUL and unattributed nodes live outside the op rows, so
    # the sum is a COVERAGE reading of the instrument, never a normaliser.
    assert node["op_wall_fraction_sum"] == pytest.approx(0.99432, abs=1e-5)
    assert [row["path"] for row in node["weight_paths"]] == [
        "expert_mul_mat_id", "dense_mul_mat", "lm_head"]


def test_node_dump_groups_the_mechanisms_the_planner_reasons_over():
    node = np.parse_node_dump(NODE_RAW)
    families = {row["family"]: row["wall_fraction"] for row in np.mechanism_shares(node)}
    assert families["moe-expert-matmul"] == pytest.approx(18626.735 / TOTAL_WALL_US)
    assert families["dense-matmul"] == pytest.approx(18348.258 / TOTAL_WALL_US)
    assert families["rms-normalization"] == pytest.approx(770.970 / TOTAL_WALL_US)
    assert families["flash-attention"] == pytest.approx(767.841 / TOTAL_WALL_US)
    # The engram gather is a real op row on this target; it must not scatter into
    # `op:<name>` singletons the way an ungrouped symbol table would.
    assert np.op_mechanism_family("GATHER_ROWS_E4M3_E8M0") == "engram-row-gather"
    assert "engram-row-gather" in {row["family"] for row in np.mechanism_shares(
        node, limit=32)}


def test_node_dump_refusals_are_strict_about_what_it_reads():
    for mutation in ({"schema": "ggml-cpu-prof/2"}, {"total_wall_us": 0.0},
                     {"ops": []}, {"ops": [{"op": "MUL_MAT", "count": 1.0,
                                            "compute_us": 1.0}]}):
        with pytest.raises(np.NodeProfileRefused):
            np.parse_node_dump(json.dumps({**NODE_DUMP, **mutation}))
    with pytest.raises(np.NodeProfileRefused):
        np.parse_node_dump("{not json")


def test_host_dump_keeps_the_two_phase_families_apart():
    host = np.parse_host_dump(HOST_RAW)
    assert host["n_eval"] == 1 and host["decode_denominator"] == "n_eval"
    assert host["phases"][0]["phase"] == "ctx.graph_compute"
    assert host["phases"][0]["family"] == "context_phase"
    assert host["phases"][0]["decode_fraction"] == pytest.approx(101154.0 / 102243.0)
    engram_input = next(row for row in host["phases"]
                        if row["phase"].endswith("llm_graph_input_dsv41_engram"))
    assert engram_input["family"] == "graph_input_class"
    assert engram_input["prefill_us_total"] == 14445.0  # a TOTAL, not per token
    assert host["phase_families_overlap"] is True


def test_host_dump_without_a_denominator_refuses_to_divide_by_it():
    # `n_eval` is filled only when llama_perf_context() ran; 0 means the per-token keys
    # are raw totals wearing a per-token name, so no fraction may be derived from them.
    host = np.parse_host_dump(json.dumps({**HOST_DUMP, "n_eval": 0}))
    assert host["decode_denominator"] == "unavailable"
    assert all(row["decode_fraction"] is None for row in host["phases"])
    with pytest.raises(np.NodeProfileRefused):
        np.parse_host_dump(json.dumps({**HOST_DUMP, "schema": "other"}))


def test_engram_dump_carries_the_fault_mix_and_its_own_measuredness():
    engram = np.parse_engram_dump(ENGRAM_RAW)
    assert engram["level"] == 2 and engram["op_level"] == 2
    assert engram["fault_source"] == "getrusage_thread"
    assert engram["op_counts_are_measured"] is True
    assert engram["fault_counts_are_measured"] is True
    assert [row["table_rows"] for row in engram["layers"]] == [384006168, 384016682]
    assert [row["engram_layer"] for row in engram["tables"]] == [0, 1]
    # This smoke accumulated no decode step span at all, so the headline ratio is
    # genuinely underivable. It is None, never 0.0.
    assert engram["host"]["n_decode_step_span"] == 0
    assert engram["gather_share_of_decode"] is None
    assert engram["majflt_per_decode_token"] == 0.0  # measured: level 2, one token


def test_unresolved_op_half_makes_its_zeros_unmeasured():
    body = {**ENGRAM_DUMP, "op_profiler": "unavailable", "op_level": 0,
            "fault_source": "none", "op": {"n_calls_unattributed": 0, "tables": []}}
    engram = np.parse_engram_dump(json.dumps(body))
    assert engram["op_counts_are_measured"] is False
    assert engram["fault_counts_are_measured"] is False
    assert engram["tables"] == []
    # Every derived rate is withheld, not reported as a zero that reads like evidence.
    assert engram["gather_share_of_decode"] is None
    assert engram["minflt_per_decode_token"] is None
    assert engram["majflt_per_decode_token"] is None


def test_section_reports_absent_dumps_without_fabricating_them(tmp_path):
    write_dumps(tmp_path, host=None, engram="{ truncated")
    body = np.section(np.read_dumps(tmp_path), build={"dir": str(tmp_path)},
                      teardown="terminated")
    assert body["status"] == "observed"
    assert body["host"]["status"] == "absent" and "dump not written" in body["host"]["reason"]
    assert body["engram"]["status"] == "absent"
    assert "host_phase_shares" not in body and "engram_fault_mix" not in body
    assert body["mechanism_shares"][0]["family"] == "moe-expert-matmul"
    assert body["limitations"] == list(np.LIMITATIONS)


def test_section_without_a_node_dump_is_absent_and_says_why(tmp_path):
    body = np.section(np.read_dumps(tmp_path / "nothing"), build={})
    assert body["status"] == "absent" and body["reason"]
    assert "mechanism_shares" not in body


def test_cache_key_is_the_perf_capture_key_plus_the_instrument_level(tmp_path):
    key = dict(anchor_commit="a" * 40, execution_digest="b" * 64,
               prompt_manifest_digest="c" * 64, scope="full", level=1)
    assert np.cached_observation(store_root=tmp_path, **key) is None
    observed = np.section({"node": np.absent("x")}, build={})
    np.retain_observation(observed, store_root=tmp_path, **key)
    assert np.cached_observation(store_root=tmp_path, **key) == observed
    for changed in ({"anchor_commit": "d" * 40}, {"execution_digest": "e" * 64},
                    {"prompt_manifest_digest": "f" * 64}, {"scope": "quarter"},
                    {"level": 2}):
        assert np.cached_observation(store_root=tmp_path, **{**key, **changed}) is None


def test_profile_loop_launches_the_sibling_with_the_instrument_and_reads_its_dumps(tmp_path):
    seen = {}

    def arm_for_env(env):
        seen["env"] = dict(env)
        return _Arm({"LD_LIBRARY_PATH": str(tmp_path / "bin"), **env}, tmp_path / "build-prof")

    def launch(template, build_dir, port, *, boot_timeout_s, resolved_recipe,
               frozen_requests, observation):
        env = dict(resolved_recipe.launch_env)
        write_dumps(Path(env[np.CPU_PROF_JSON_ENV]).parent)
        seen["launched"] = (str(build_dir), port, frozen_requests)
        observation.append({"teardown": "terminated"})
        return 12.8

    body = np.profile_loop(arm_for_env, _Prompts(), store_root=tmp_path,
                           build={"dir": str(tmp_path / "build-prof")}, level=2,
                           launch=launch)
    assert seen["env"][np.CPU_PROF_GATE_ENV] == "1"
    assert seen["env"][np.ENGRAM_LEVEL_ENV] == "2"
    assert seen["launched"][0] == str(tmp_path / "build-prof")
    assert seen["launched"][2] == (("p0", b'{"prompt":"x"}'),)
    assert body["status"] == "observed" and body["teardown"] == "terminated"
    assert body["engram_fault_mix"]["fault_counts_are_measured"] is True
    assert body["host_phase_shares"][0]["phase"] == "ctx.graph_compute"


def test_profile_loop_refuses_an_arm_that_dropped_the_instrument(tmp_path):
    def arm_for_env(env):
        return _Arm({"LD_LIBRARY_PATH": "/x"}, tmp_path)

    with pytest.raises(np.NodeProfileRefused, match="instrument environment"):
        np.profile_loop(arm_for_env, _Prompts(), store_root=tmp_path, build={},
                        launch=lambda *a, **k: None)


def test_cpu_arm_extra_env_never_reaches_the_measured_arm():
    """`_cpu_arm` without `extra_env` is byte-identical to today's measured rebind."""
    from . import run

    source = Path(run.__file__).read_text(encoding="utf-8")
    assert "def _cpu_arm(original, build: Path, *, extra_env: dict | None = None)" in source
    assert source.count("extra_env=env") == 1  # only the node-profile sibling passes it
