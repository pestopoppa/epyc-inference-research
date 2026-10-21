"""`--serving-calibration-only` (operator 2026-10-21, DS41-1017 floor window).

A scheduled floor window must calibrate the matched serving floor and then END: no
reprofile, no planner, no proposal, terminal 'stopped' so serial_run accepts 0 of N
iterations and stops. Mirrors the held-out calibration-only branch test.
"""
from pathlib import Path

HERE = Path(__file__).parent
RUN = (HERE / "run.py").read_text()
ROSTER = (HERE / "serial_roster.py").read_text()


def test_flag_is_declared_and_refused_without_explicit_cpu_calibration():
    assert '"--serving-calibration-only", action="store_true"' in RUN
    rule = RUN.index("if args.serving_calibration_only and (")
    body = RUN[rule:rule + 400]
    assert "args.cpu_calibrate_serving is None" in body
    assert "not args.cpu_serving_launch" in body
    assert "args.validate_source_continuation" in body
    assert "parser.error(" in body


def test_calibration_only_skips_reprofile_and_records_not_collected():
    branch = RUN.index("            if args.serving_calibration_only:\n                cpu_profile_observation")
    nxt = RUN.index("reprofile()", branch)
    window = RUN[branch:nxt]
    assert 'cpu_profile_observation.update(status="not_collected"' in window
    assert 'node_profile_observation.update(status="not_collected"' in window
    assert "elif screen_confirmation is None and not args.validate_source_continuation:" in window


def test_calibration_only_branch_precedes_pool_and_requests_stop():
    end = RUN.index("        publish_held_claims()\n        elapsed = time.time() - started")
    branch = RUN.rfind(
        "            if args.heldout_calibration_only or args.serving_calibration_only:", 0, end)
    assert branch > 0
    seg = RUN[branch:end]
    assert "pooled = pool.PoolResult(outcomes=[]" in seg
    first_else = seg.index("elif args.validate_source_continuation:")
    assert 'stopping["asked"] = True' in seg[:first_else]
    assert "else:\n                pooled = run_pooled()" in seg


def test_terminal_is_stopped_when_stop_was_asked():
    assert 'terminal="stopped" if should_stop() else "complete"' in RUN


def test_serial_roster_admits_the_floor_window_switches():
    start = ROSTER.index("switches = {")
    block = ROSTER[start:ROSTER.index("}", start)]
    assert '"--require-serving-floor"' in block
    assert '"--serving-calibration-only"' in block


def test_calibration_only_also_calibrates_an_opted_in_longctx_surface():
    """DS41-W8a: the long-context surface's own slot + floor (longctx.Surface) are
    calibrated in the same calibration-only window, before the 0-outcome exit, through
    the run's measurement window and stop-aware calibrator."""
    end = RUN.index("            if args.heldout_calibration_only or args.serving_calibration_only:")
    start = RUN.rindex("            if (args.serving_calibration_only and longctx_surface is not None", 0, end)
    seg = RUN[start:end]
    assert "and not should_stop()" in seg
    assert "longctx_surface.launch_for(direct_launch), anchor_build[0]" in seg
    assert "with cpu_measurement_window():\n                    long_launch = longctx_surface.ensure_slot(long_arm)" in seg
    assert "calibrate=measured_serving_calibrate" in seg
