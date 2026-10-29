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
    # CH-27: a floor window never reopens a floor measured on another anchor, and the
    # long-context floor goes through the execution-equivalence registry like the serving one
    assert "require_anchor=True" in seg
    assert "_registry_floor_reading(long_arm.template, long_arm" in seg
    assert "_registry_register(long_arm.template, long_arm, row, written=path" in seg


def test_store_anchor_is_admitted_on_an_experimental_lane():
    """CH-27 (2026-10-23): a kernel-store image (--floor-registry, no provenance.json, no
    IDENTITY.json -- the store is immutable) must be admissible as the anchor of an experimental
    CPU serving lane. The in-build IDENTITY receipt check is skipped for it; the store attestation
    (commit ancestor-or-equal of the champion head) and the clean-source rule still apply."""
    start = RUN.index("        store_anchor = (_FLOOR_REGISTRY is not None")
    seg = RUN[start:RUN.index("    if direct_launch is not None and not args.dry_run:", start)]
    assert "experimental_identity=experimental and not source_authoring" in seg
    assert "floor_registry.store_build_commit_is_descendant(" in seg
    assert 'if experimental and _git(args.worktree, "status", "--porcelain"):' in seg


def test_store_image_is_admitted_as_the_exact_champion_of_record_build():
    """CH-27: a plain champion advance keeps the store's COR; with --floor-registry its kernel-
    store image (no provenance/IDENTITY receipt) is the exact COR arm iff VERSION.txt names
    exactly that commit. Both the dry-run and the live verification go through the helper."""
    helper = RUN[RUN.index("def _verify_cor_build("):RUN.index("def _dry_run_accumulator_and_resume(")]
    assert "floor_registry.is_store_build(path)" in helper
    assert "named != serial_run.full_commit(worktree, commit)" in helper
    assert "serial_run.verify_exact_anchor(path, worktree, commit" in helper
    assert RUN.count("_verify_cor_build(") == 3          # definition + dry + live
    assert "serial_run.verify_exact_anchor(args.cor_build" not in RUN
