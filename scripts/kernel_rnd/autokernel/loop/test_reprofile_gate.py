"""AK-EVAL-KEEPTIME: conditional per-keep reprofile. Synthetic diffs; no profiler runs."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from . import cpu_profile, cpu_screen, keep_timing, loop, reprofile_gate, run

HOTSPOTS = [{"dso": "libggml-cpu.so", "symbol": "ggml_vec_dot_q4_K_q8_K", "period": 9},
            {"dso": "libggml-cpu.so", "symbol": "ggml_compute_forward_mul_mat", "period": 3}]
DISJOINT = """diff --git a/ggml/src/ggml-cpu/ops.cpp b/ggml/src/ggml-cpu/ops.cpp
@@ -10,3 +10,3 @@ static void ggml_compute_forward_rope_f32(
-    const int n_dims = 4;
+    const int n_dims = 8;
"""
OVERLAP = DISJOINT.replace("ggml_compute_forward_rope_f32", "ggml_vec_dot_q4_K_q8_K")


def test_skip_reason_is_conservative():
    assert reprofile_gate.skip_reason(DISJOINT, HOTSPOTS) is not None
    assert reprofile_gate.skip_reason(OVERLAP, HOTSPOTS) is None
    assert reprofile_gate.skip_reason("", HOTSPOTS) is None
    assert reprofile_gate.skip_reason(DISJOINT, []) is None
    assert reprofile_gate.hotspot_name("libggml-cpu.so:ns::foo<int>(float)") == "foo"
    assert reprofile_gate.hotspot_name({"symbol": "ggml_vec_dot_q8_0_q8_0.cold"}) == "ggml_vec_dot_q8_0_q8_0"


def _reprofile(diff, observation, timer):
    profiled = []

    def profile_loop(*_a, **_k):
        profiled.append(1)
        return {"hotspots": HOTSPOTS, "record": "r"}
    args = SimpleNamespace(worktree=Path("/wt"), store=Path("/store"), cpu_profiler=None)
    with mock.patch.object(run, "_git", return_value=diff), \
            mock.patch.object(run, "_cpu_arm", return_value=SimpleNamespace(execution_digest="d")), \
            mock.patch.object(cpu_profile, "profile_loop", side_effect=profile_loop):
        run._reprofile(cpu_launch=lambda: True, cpu_profile_observation=observation,
            direct_launch=lambda: None, anchor_build=[Path("/a")], resumed=None, args=args,
            current_anchor_commit=["abc123"], manifest=lambda: None, screen_state=None,
            node_reprofile=lambda _arm: None, longctx_reprofile=lambda: None,
            publish=lambda *a, **k: None, latest=None, cpu_measurement_window=nullcontext,
            selected_identity=None, resolved_campaign=None, gpu_profile_observation={},
            frozen_requests=None, gpu_long_context_hook=[None], hotspot_rows=[], pp=0, tg=0,
            keep_timer=[timer])
    return profiled


def test_disjoint_keep_skips_with_reason_and_overlapping_keep_reprofiles():
    timer = keep_timing.KeepTimer(head="abc123")
    observation = {"status": "observed", "hotspots": HOTSPOTS}
    assert _reprofile(DISJOINT, observation, timer) == []
    skipped = timer.to_dict()["stages"]
    assert skipped[0]["stage"] == "reprofile_cpu" and skipped[0]["skipped"] is True
    assert "hotspot" in skipped[0]["reason"] and observation["status"] == "observed"
    timer = keep_timing.KeepTimer(head="abc123")
    assert _reprofile(OVERLAP, {"status": "observed", "hotspots": HOTSPOTS}, timer) == [1]
    assert timer.to_dict()["stages"] == []


def test_aborted_keep_carries_its_stage_timing_on_the_abort_evidence():
    hypothesis = loop.Hypothesis("k", "statement", "falsifier", "pp512", "ggml/src/x.c")
    planner = mock.Mock()
    planner.propose.return_value = hypothesis
    planner.author.return_value = ("ggml/src/x.c",)
    critic = cpu_screen.RetainedCritic({"hypothesis": hypothesis.to_dict(), "paths": ["ggml/src/x.c"]})
    comparison = run.bench.Comparison(surface="pp512", anchor_samples=[100.0] * 5,
        candidate_samples=[110.0] * 5, effect=0.1, estimator="median_over_median", pairs=5,
        noise_floor_pct=1.0, residency={})
    aborted = loop.RunAborted("anchor guard")
    aborted.keep_path_timing = {"schema": keep_timing.SCHEMA, "stages": []}
    try:
        loop.iterate(planner=planner, critic=critic, context={},
            measure=lambda _h, _p: comparison,
            gate=lambda _h, _p: (True, [run.gates.Verdict("oracle", True)]),
            commit=mock.Mock(side_effect=aborted))
    except loop.RunAborted as exc:
        assert exc.inflight_keep["keep_path_timing"]["schema"] == keep_timing.SCHEMA
    else:
        raise AssertionError("abort must propagate")
