"""Regression checks for the run argument and phase-monitor split."""
from pathlib import Path

import pytest

from . import run, run_args


def test_parser_keeps_planner_only_anchor_exception():
    parser = run_args.build_parser(["--planner-only"], description=run.__doc__, run_module=run)
    args = parser.parse_args(["--planner-only", "--worktree", "/tmp/work", "--store", "/tmp/store"])
    assert args.worktree == Path("/tmp/work")
    assert args.anchor_build is None
    assert args.amdahl_prune == run.amdahl.DEFAULT_MODE
    assert args.gpu_window == "off"
    parser = run_args.build_parser([], description=run.__doc__, run_module=run)
    with pytest.raises(SystemExit):
        parser.parse_args(["--worktree", "/tmp/work", "--store", "/tmp/store"])


def test_recorded_gate_failure_clears_stale_success():
    from types import SimpleNamespace

    numerics = {"candidate": {"passed": True}}
    hypothesis = SimpleNamespace(mechanism_id="candidate")

    def failed_gate(_hypothesis, _paths):
        raise RuntimeError("gate failed")

    with pytest.raises(RuntimeError, match="gate failed"):
        run._recorded_gate(numerics, failed_gate)(hypothesis, [])
    assert "candidate" not in numerics


def test_stop_signal_latches_and_is_seen_at_boundary(tmp_path):
    stopping = {"asked": False}
    assert not run._stop_asked(stopping, tmp_path)
    run._ask_stop(stopping, 15, None)
    assert run._stop_asked(stopping, tmp_path)


def test_graceful_calibration_stop_only_swallows_owned_stop():
    finished = []
    with pytest.raises(run.loop.ActorStopped):
        with run._graceful_calibration_stop([False], lambda: finished.append(True)):
            raise run.loop.ActorStopped()
    with run._graceful_calibration_stop([True], lambda: finished.append(True)):
        raise run.loop.ActorStopped()
    assert finished == [True]


def test_anchor_gate_caches_only_successful_calls():
    from types import SimpleNamespace

    builds, cache, calls = [Path('/tmp/anchor')], {}, []
    verdict = SimpleNamespace(gate='oracle', passed=True)

    def correctness(build, **kwargs):
        calls.append((build, kwargs))
        if len(calls) == 1:
            raise RuntimeError('oracle unavailable')
        return verdict

    with pytest.raises(RuntimeError, match='oracle unavailable'):
        run._anchor_gate(builds, cache, correctness, 'op')
    assert run._anchor_gate(builds, cache, correctness, 'op') is verdict
    assert run._anchor_gate(builds, cache, correctness, 'op') is verdict
    assert len(calls) == 2


def test_reset_retained_does_not_reset_if_archive_fails(monkeypatch):
    from types import SimpleNamespace

    reset = []
    def fail(*args, **kwargs):
        raise RuntimeError('archive unavailable')
    monkeypatch.setattr(run.archive, 'retain_patch', fail)
    monkeypatch.setattr(run.pool, 'reset_to_champion', lambda *args, **kwargs: reset.append(True))
    worker = SimpleNamespace(worktree=Path('/tmp/lane'), name='lane')
    args = SimpleNamespace(store=Path('/tmp/store'), worktree=Path('/tmp/tree'), champion_branch='main')
    with pytest.raises(RuntimeError, match='archive unavailable'):
        run._reset_retained(args, worker)
    assert not reset


def test_confirm_measure_uses_current_comparison_callback():
    from types import SimpleNamespace

    observed = []
    current = [lambda *args, **kwargs: observed.append('old')]
    args = SimpleNamespace(confirm_model=Path('/tmp/model'), confirm_pairs=1)
    measure = run._confirm_measure(args, [Path('/tmp/anchor')],
                                   lambda *a, **kw: current[0](*a, **kw),
                                   SimpleNamespace(build_dir=Path('/tmp/candidate')))
    current[0] = lambda *args, **kwargs: observed.append('new')
    measure('tg128', 1.0)
    assert observed == ['new']



def test_bundle_judge_reads_current_launch(monkeypatch, tmp_path):
    from types import SimpleNamespace

    current = [object()]
    seen = []
    def arm(launch, build):
        seen.append(launch)
        return SimpleNamespace(launch_env={})
    monkeypatch.setattr(run, '_cpu_arm', arm)
    monkeypatch.setattr(run.gates, 'pinned_production_reference', lambda *a, **kw: tmp_path)
    monkeypatch.setattr(run.gates, 'ppl_contract_bundle_gate', lambda *a, **kw: None)
    args = SimpleNamespace(store=tmp_path, model=tmp_path / 'model')
    run._ppl_contract_judge_bundle(args, lambda: current[0], [tmp_path], '80-87', tmp_path)
    current[0] = object()
    run._ppl_contract_judge_bundle(args, lambda: current[0], [tmp_path], '80-87', tmp_path)
    assert seen[0] is not seen[1] and seen[1] is current[0]


def test_long_context_runtime_gate_fails_closed_on_measurement_error(monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError('measurement unavailable')
    monkeypatch.setattr(run, '_serving_comparison', failed)
    row = run._longctx_runtime_gate(None, [], None, lambda: None, None)
    assert row['passed'] is False and 'measurement unavailable' in row['reason']


def test_long_context_compare_reads_current_launch(monkeypatch):
    from types import SimpleNamespace

    current, seen = [object()], []
    surface = SimpleNamespace(launch_for=lambda launch: seen.append(launch))
    arm = SimpleNamespace(execution_digest='digest', template=SimpleNamespace(model='model'))
    monkeypatch.setattr(run, '_cpu_arm', lambda *args: arm)
    monkeypatch.setattr(run, '_serving_comparison', lambda *args, **kwargs: None)
    args = SimpleNamespace(serving_pairs=1)
    kwargs = dict(longctx_surface=lambda: surface, direct_launch=lambda: current[0],
                  anchor_build=lambda: ['anchor'], experimental=lambda: False,
                  cpu_measurement_window=lambda: (lambda: None), args=lambda: args,
                  retain_measured_source_comparison=lambda: None)
    run._longctx_compare(None, **kwargs)
    current[0] = object()
    run._longctx_compare(None, **kwargs)
    assert seen[0] is not seen[1] and seen[1] is current[0]


def test_process_guard_uninstalls_even_if_sweep_fails(monkeypatch):
    from types import SimpleNamespace

    removed = []
    def failed(_phase):
        raise OSError('sweep failed')
    guard = SimpleNamespace(sweep_stale=failed)
    monkeypatch.setattr(run.procguard, 'uninstall', removed.append)
    with pytest.raises(OSError, match='sweep failed'):
        run._end_proc_guard(guard)
    assert removed == [guard]


def test_runtime_progress_failure_is_observational(capsys):
    status = [None]
    run._report_runtime_progress(status, {}, [None], [None], lambda *a, **kw: None, [], [])
    assert status == [None]
    assert 'runtime progress unavailable' in capsys.readouterr().err


def test_long_context_keep_gate_fails_closed_on_measurement_error(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(run.longctx, 'attention_route', lambda _hypothesis: False)
    def failed(_build):
        raise ValueError('measurement unavailable')
    row = run._longctx_keep_gate(None, [], failed, SimpleNamespace(build_dir=Path('/tmp/build')),
                                 object(), None)
    assert row['passed'] is False and 'measurement unavailable' in row['reason']


def test_stale_rebaseline_uses_cpu_comparison_for_a_serving_launch():
    from types import SimpleNamespace

    calls = []
    def compare(a, b, **kwargs):
        calls.append((a, b, kwargs))
        return 'comparison'
    result = run._measure_rebaseline(compare, [Path('/tmp/cor')], [Path('/tmp/anchor')],
                                     object(), None, SimpleNamespace(), 0, 0, None, 'tg128', None)
    assert result == 'comparison'
    assert calls == [(Path('/tmp/cor'), Path('/tmp/anchor'), {'rebind_feedback': False})]



def test_cpu_acquisition_latches_even_if_marker_fails(monkeypatch):
    acquired = [False]
    def fail(*args):
        raise OSError('marker unavailable')
    monkeypatch.setattr(run, '_publish_cpu_native_acquired', fail)
    with pytest.raises(OSError, match='marker unavailable'):
        run._native_cpu_acquired(acquired, None, None, None)
    assert acquired == [True]


def test_baseline_build_refuses_stale_source_before_compilation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    called = []
    monkeypatch.setattr(run, '_git', lambda *args: 'old-head')
    verdict = run._build_baseline(lambda *a, **kw: called.append(True), SimpleNamespace(),
                                  1, '80-87', tmp_path, 'new-head')
    assert not verdict.passed and not called


def test_cpu_claim_acquisition_refusal_propagates(monkeypatch):
    from contextlib import ExitStack, contextmanager

    @contextmanager
    def refused(*args, **kwargs):
        raise RuntimeError('claim refused')
        yield
    monkeypatch.setattr(run.claim, 'hold_cpu', refused)
    with ExitStack() as stack:
        with pytest.raises(RuntimeError, match='claim refused'):
            run._hold_cpu_claim(None, stack, {}, '80-87')


def test_champion_build_keeps_requested_tools_for_direct_launch(tmp_path):
    from types import SimpleNamespace

    calls = []
    args = SimpleNamespace(worktree=tmp_path / 'source')
    run._build_champion(lambda *a, **kw: calls.append(kw), args,
                        run.build_recipe.NATIVE_CPU_RECIPE, 4, '80-87', lambda: object(),
                        tmp_path / 'build', ('custom-tool',))
    targets = calls[0]['targets']
    assert 'custom-tool' in targets and set(run.gates.PROMOTION_TARGETS) <= set(targets)


def test_candidate_profile_failure_does_not_refuse_comparison(monkeypatch, tmp_path):
    from types import SimpleNamespace

    def fail(*args):
        raise OSError('profile source unavailable')
    monkeypatch.setattr(run.archive, '_git', fail)
    monkeypatch.setattr(run.serving_beliefs, 'annotate', lambda _row, fields: fields)
    row = {}
    run._record_candidate_serving_profile(['anchor'], None, None, None, None, [None], {},
                                          SimpleNamespace(row=row),
                                          SimpleNamespace(worktree=tmp_path), None)
    assert row['candidate_serving_profile']['status'] == 'unavailable'


def test_runtime_adoption_receipt_failure_is_reported_without_raising(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from . import runtime_arms

    def fail(**kwargs):
        raise OSError('receipt unavailable')
    monkeypatch.setattr(runtime_arms, 'adoption_receipt', fail)
    preparation = {}
    run._record_runtime_adoption(None, [None], SimpleNamespace(store=tmp_path), None, None,
                                 [None], 'epoch', 'measurement', ['anchor'], None, None, False,
                                 [], preparation, None, None,
                                 SimpleNamespace(row={}), SimpleNamespace(to_dict=dict), None)
    assert 'receipt unavailable' in preparation['adoption_receipt_error']


def test_counter_phase_charges_even_when_wrapped_function_raises():
    samples, charges = [], []
    class Counter:
        def sample(self, boundary):
            samples.append(boundary)
    @run._counter_phase('measure', [Counter()], [lambda *args: charges.append(args)])
    def failing():
        raise RuntimeError('expected')
    with pytest.raises(RuntimeError, match='expected'):
        failing()
    assert samples == ['measure:start', 'measure:end']
    assert len(charges) == 1 and charges[0][0] == 'measure' and charges[0][1] >= 0


def test_long_identity_targets_run_inside_measurement_window(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    events = []
    @contextmanager
    def window():
        events.append('enter')
        try:
            yield
        finally:
            events.append('exit')
    monkeypatch.setattr(run, '_cpu_arm', lambda launch, build: (launch, build))
    surface = SimpleNamespace(launch_for=lambda launch: launch,
                              identity_targets=lambda a, c: events.append(('identity', a, c)))
    run._long_identity_targets_impl(surface, 'launch', window, 'a', 'c')
    assert events == ['enter', ('identity', ('launch', 'a'), ('launch', 'c')), 'exit']


def test_ancestor_check_uses_the_selected_worktree(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []
    def fake(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(run.subprocess, 'run', fake)
    assert not run._run_is_ancestor(SimpleNamespace(worktree=tmp_path), 'a', 'b')
    assert calls == [['git', '-C', str(tmp_path), 'merge-base', '--is-ancestor', 'a', 'b']]


def test_patch_retention_preserves_lane_and_mechanism(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(run.archive, 'retain_patch', lambda *a, **kw: calls.append((a, kw)))
    run._keep_the_diff_impl(SimpleNamespace(store=tmp_path),
                            SimpleNamespace(worktree=tmp_path / 'lane', name='lane-1'),
                            SimpleNamespace(mechanism_id='mechanism'))
    assert calls == [((tmp_path, tmp_path / 'lane'), {'lane': 'lane-1', 'mechanism_id': 'mechanism'})]


def test_owned_gpu_window_does_not_require_status_publisher():
    from types import SimpleNamespace

    window = object()
    def unavailable():
        raise AssertionError('status publisher is not ready')
    local = SimpleNamespace(compute=lambda: window)
    assert run._cpu_measurement_window_impl(local, unavailable, None, None, unavailable,
                                            unavailable, None, None, None) is window


def test_accumulator_status_uses_current_floor_and_bundle(monkeypatch, tmp_path):
    from types import SimpleNamespace

    policy = SimpleNamespace(fire_multiple=2, every_keeps=4, fire_threshold_pct=lambda floor: floor * 2)
    bundle = SimpleNamespace(tip="tip", keeps=["one"], compounded_bench_pct=3.0,
                             keeps_since_serving_gate=1)
    monkeypatch.setattr(run.accumulate, "effective_measurement_validity",
                        lambda *_: run.accumulate.MEASUREMENT_CURRENT)
    monkeypatch.setattr(run.accumulate, "gate_trigger", lambda *_args, **_kw: None)
    args = SimpleNamespace(store=tmp_path)
    row = run._accumulator_state(lambda: object(), lambda: 2.0, policy, [bundle], args, ["cor"],
                                 lambda: "measured", lambda: None, lambda: "percent", [None])
    assert row["fire_threshold_pct"] == 4.0
    assert row["compounded_bench_pct"] == 3.0
    assert row["progress_fraction"] == 0.75


def test_node_profile_disabled_records_reason(monkeypatch):
    import inspect
    from types import SimpleNamespace
    from . import node_profile

    monkeypatch.setattr(node_profile, "refuse_instrumented_measurement", lambda arm: None)
    values = {name: None for name in inspect.signature(run._node_reprofile).parameters
              if name != "profile_arm"}
    observation = {}
    values.update(node_profile_observation=observation, args=SimpleNamespace(node_profile=False))
    run._node_reprofile(object(), **values)
    assert observation == {"status": "not_collected",
                           "reason": "instrumented sibling profiling disabled"}


def test_longctx_profile_without_surface_does_nothing():
    import inspect

    values = {name: None for name in inspect.signature(run._longctx_reprofile).parameters}
    assert run._longctx_reprofile(**values) is None


def test_accumulator_guard_records_errors_after_keep(monkeypatch, tmp_path):
    from types import SimpleNamespace

    writes = []
    monkeypatch.setattr(run.status, 'write_json', lambda *args, **kwargs: writes.append((args, kwargs)))

    def fail(_mechanism, _owner):
        raise ValueError('unavailable')

    with pytest.raises(run.loop.RunAborted, match="unresolved accumulator"):
        run._accumulate_after_keep_guard('candidate', object(), serving_recipe=object(),
                                        _accumulate_after_keep=fail,
                                        args=SimpleNamespace(store=tmp_path))
    assert writes[0][0][2]['mechanism_id'] == 'candidate'
    assert 'unavailable' in writes[0][0][2]['error']


def test_node_profile_reuses_cached_observation(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace
    from . import node_profile

    monkeypatch.setattr(node_profile, 'refuse_instrumented_measurement', lambda arm: None)
    monkeypatch.setattr(node_profile, 'cached_observation', lambda **kwargs: {'status': 'cached'})
    values = {name: None for name in inspect.signature(run._node_reprofile).parameters
              if name != 'profile_arm'}
    observation = {}
    values.update(node_profile_observation=observation,
                  args=SimpleNamespace(node_profile=True, node_profile_level='op', store=tmp_path),
                  screen_state=None, current_anchor_commit=['abc'],
                  manifest=lambda: SimpleNamespace(digest='request'))
    run._node_reprofile(SimpleNamespace(execution_digest='exec'), **values)
    assert observation == {'status': 'cached', 'anchor_commit': 'abc'}


def test_experimental_headline_never_runs_production(monkeypatch):
    import inspect

    def unexpected(**kwargs):
        raise AssertionError('production refresh should not run')

    monkeypatch.setattr(run.production, 'refresh', unexpected)
    values = {name: None for name in inspect.signature(run._publish_headline).parameters}
    values['experimental'] = True
    assert run._publish_headline(**values) is None


def test_ppl_tools_build_reuses_complete_slot_without_build(tmp_path):
    import inspect

    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for tool in run.gates.PPL_CONTRACT_TOOL_TARGETS:
        (bindir / tool).touch()
    values = {name: None for name in inspect.signature(run._ppl_contract_tools_build).parameters
              if name not in ('slot', 'commit')}
    assert run._ppl_contract_tools_build(tmp_path, 'commit', **values) == tmp_path


def test_ppl_tools_build_refuses_slot_without_objects(monkeypatch, tmp_path):
    import inspect

    monkeypatch.setattr(run.anchor_integrity, 'object_digest', lambda slot: None)
    values = {name: None for name in inspect.signature(run._ppl_contract_tools_build).parameters
              if name not in ('slot', 'commit')}
    with pytest.raises(ValueError, match='no library objects'):
        run._ppl_contract_tools_build(tmp_path, 'commit', **values)


def test_ppl_bisect_tip_reuses_anchor_without_build(tmp_path):
    import inspect

    values = {name: None for name in inspect.signature(run._ppl_contract_bisect_build).parameters
              if name != 'commit'}
    values.update(current_anchor_commit=['tip'], ppl_contract_anchor_for_gate=lambda: tmp_path)
    assert run._ppl_contract_bisect_build('tip', **values) == tmp_path


def test_ppl_fold_error_is_a_refusal(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace

    def fail(*args):
        raise RuntimeError('repository unavailable')

    monkeypatch.setattr(run, '_git', fail)
    values = {name: None for name in inspect.signature(run._ppl_contract_fold_check).parameters
              if name != 'tip_commit'}
    values.update(args=SimpleNamespace(worktree=tmp_path), cor_commit=['cor'])
    verdict = run._ppl_contract_fold_check('tip', **values)
    assert not verdict.passed
    assert 'repository unavailable' in verdict.reason


def test_ppl_eviction_holds_unavailable_evidence(monkeypatch):
    import inspect

    monkeypatch.setattr(run.gates, 'ppl_contract_failure_class', lambda quality: 'unavailable')
    values = {name: None for name in inspect.signature(run._ppl_contract_evict_culprit).parameters
              if name not in ('tip_commit', 'quality')}
    result = run._ppl_contract_evict_culprit('tip', object(), **values)
    assert result['result'] == 'held_unavailable'


def test_reprofile_handles_profiler_unavailability(monkeypatch, tmp_path):
    import inspect
    from contextlib import nullcontext
    from types import SimpleNamespace

    def fail(*args, **kwargs):
        raise run.hotspots.ProfileFailed('unavailable')

    monkeypatch.setattr(run.hotspots, 'profile', fail)
    values = {name: None for name in inspect.signature(run._reprofile).parameters}
    values.update(cpu_launch=lambda: None, direct_launch=lambda: None, anchor_build=[tmp_path],
                  args=SimpleNamespace(model=tmp_path / 'model'),
                  cpu_measurement_window=nullcontext, hotspot_rows=['previous'])
    run._reprofile(**values)
    assert values['hotspot_rows'] == ['previous']


def test_keep_diff_preserves_lineage_capture(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(run.archive, 'retain_patch', lambda *a, **kw: tmp_path / 'patch')
    monkeypatch.setattr(run.lineage_capture, 'capture_at_keep',
                        lambda *a, **kw: calls.append((a, kw)))
    args = SimpleNamespace(store=tmp_path, lineage_capture=True)
    worker = SimpleNamespace(worktree=tmp_path, name='lane')
    result = run._keep_the_diff_impl(args, worker, SimpleNamespace(mechanism_id='idea'))
    assert result == tmp_path / 'patch'
    assert calls[0][1]['enabled'] is True


def test_publish_contains_scope_gap_failure(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace
    from . import keep_screen

    def fail(*args):
        raise ValueError('gap unavailable')

    writes = []
    monkeypatch.setattr(run.status, 'stagnation_signal', lambda attempts: None)
    monkeypatch.setattr(run.status, 'write', lambda *args, **kwargs: writes.append(kwargs))
    monkeypatch.setattr(keep_screen, 'project', lambda *args, **kwargs: {})
    monkeypatch.setattr(run, '_git', lambda *args: 'head')
    values = {name: None for name, param in inspect.signature(run._publish).parameters.items()
              if param.kind == param.KEYWORD_ONLY}
    values.update(direct_launch=lambda: None, floor=lambda: None,
                  uncovered_gap_state=[None], scope_gap_publisher=SimpleNamespace(update=fail),
                  scope_gap_error=[None], accumulator_state=lambda: None,
                  args=SimpleNamespace(store=tmp_path, surface='s', pairs=1, serving_pairs=1,
                      model='model', iterations=1, worktree=tmp_path, out=None,
                      no_measurement_alarm_n=1, keep_rate_alarm_window=1),
                  current_anchor_commit=['anchor'], runtime_status=[None], anchor_guard_seen=[],
                  pending_view=[None], live_tail=[None], actor_health=lambda outcomes: {},
                  history_view=[None], scratch_registry=[None])
    run._publish('running', gpu={}, **values)
    assert writes[0]['scope_gap_error']['message'] == 'gap unavailable'


def test_gate_factory_reads_live_runtime_state():
    import inspect
    from types import SimpleNamespace

    values = {name: None for name in inspect.signature(run._gate_for).parameters
              if name != 'worker'}
    enabled = [False]
    values.update(recorded_gate=lambda gate: gate, screen_state=None,
                  runtime_enabled=lambda: enabled[0], runtime_probe_enabled=False,
                  runtime_preparation={'reason': 'not ready'}, direct_launch=lambda: None,
                  cpu_launch=lambda: None)
    gate = run._gate_for(object(), **values)
    hypothesis = SimpleNamespace(runtime_pair=object())
    assert gate(hypothesis, [])[1][0].gate == 'runtime_preparation'
    enabled[0] = True
    assert gate(hypothesis, [])[1][0].gate == 'runtime_treatment'


def test_measure_factory_disables_exhausted_runtime(monkeypatch):
    import inspect
    from types import SimpleNamespace
    from . import runtime_calibration

    enabled = [True]
    class Owner:
        def compare(self, pair):
            raise runtime_calibration.RuntimeLaunchBudgetExhausted('spent')

    monkeypatch.setattr(run, '_serving_comparison', lambda fn, *args, **kwargs: fn())
    values = {name: None for name in inspect.signature(run._measure_for).parameters
              if name != 'worker'}
    preparation = {}
    values.update(disable_runtime=lambda: enabled.__setitem__(0, False),
                  runtime_enabled=lambda: enabled[0], runtime_keep_grade=False,
                  runtime_owner=[Owner()], runtime_preparation=preparation,
                  report_runtime_progress=lambda: None)
    measure = run._measure_for(object(), **values)
    with pytest.raises(run.loop.TailRefused, match='budget ended'):
        measure(SimpleNamespace(runtime_pair=object()), [])
    assert enabled == [False]
    assert preparation['status'] == 'budget_exhausted'


def test_pooled_driver_uses_live_callback_state(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace

    from . import phase_journal

    captured, closed = {}, []
    monkeypatch.setattr(run.accumulate, '_surface_paths', lambda *args: ('unused', 'journal'))
    monkeypatch.setattr(run.accumulate.journal, 'Journal', lambda *args: object())
    monkeypatch.setattr(phase_journal, 'PhaseJournal', lambda journal: SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(run.resume_mod, 'target_identity', lambda **kwargs: 'target')
    monkeypatch.setattr(run.pool, 'provision', lambda *args, **kwargs: [])
    monkeypatch.setattr(run.pool, 'drive', lambda **kwargs: captured.update(kwargs) or 'result')
    monkeypatch.setattr(run, '_sandbox_scratch', lambda *args: None)
    monkeypatch.setattr(run, '_gate_wrappers', lambda *args: [])
    monkeypatch.setattr(run, '_tail_split_enabled', lambda args: False)
    values = {name: (lambda: None) for name in inspect.signature(run._run_pooled).parameters
              if name != 'set_runtime_selection'}
    args = SimpleNamespace(surface='s', model='model', resume='off', workers=1,
                           worktree=tmp_path, champion_branch='main', worker_root=tmp_path,
                           worker_build_root=tmp_path, iterations=1,
                           hypothesis_author_attempts=1, store=tmp_path, bundle_surface=None)
    holder = [[None]]
    values.update(args=lambda: args, current_anchor_commit=lambda: ['head'],
                  runtime_enabled=lambda: False, scratch_registry=lambda: [None],
                  author_plan=lambda: SimpleNamespace(panel=False),
                  cpu_window_ref=lambda: [None], live_tail=lambda: holder[0])
    assert run._run_pooled(set_runtime_selection=lambda selected: None, **values) == 'result'
    assert closed == [True]
    assert captured['phase_journal'] is not None
    holder[0] = [None]
    captured['on_tail_ready']('tail')
    assert holder[0] == ['tail']


def test_accumulator_implementation_propagates_unavailable_head(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace

    def fail(*args):
        raise RuntimeError('head unavailable')

    monkeypatch.setattr(run, '_git', fail)
    values = {name: (lambda: None) for name in inspect.signature(run._accumulate_after_keep_impl).parameters
              if name not in ('mechanism_id', 'phase_owner')}
    values['args'] = lambda: SimpleNamespace(worktree=tmp_path)
    with pytest.raises(RuntimeError, match='head unavailable'):
        run._accumulate_after_keep_impl('idea', object(), **values)


def test_verify_anchor_preserves_floor_override(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace

    observed = {}
    monkeypatch.setattr(run, '_git', lambda *args: 'head')
    monkeypatch.setattr(run.anchor, 'verify', lambda **kwargs: observed.update(kwargs))
    values = {name: (lambda: None) for name in inspect.signature(run._verify_anchor).parameters
              if name != 'guard_floor'}
    values.update(args=lambda: SimpleNamespace(worktree=tmp_path, store=tmp_path),
                  direct_launch=lambda: None, cpu_launch=lambda: None,
                  anchor_build=lambda: [tmp_path], floor=lambda: 2.0)
    run._verify_anchor(guard_floor=3.0, **values)
    assert observed['champion_commit'] == 'head'
    assert observed['noise_floor_pct'] == 3.0


def test_promote_anchor_keeps_committed_head_on_guard_failure(monkeypatch, tmp_path):
    import inspect
    from contextlib import nullcontext
    from types import SimpleNamespace

    monkeypatch.setattr(run, '_git', lambda *args: 'head')
    monkeypatch.setattr(run.gates, 'ppl_contract_anchor_obligated', lambda store: False)
    monkeypatch.setattr(run.pool, 'promote_anchor', lambda *args, **kwargs: tmp_path / 'new')
    def fail(**kwargs):
        raise run.loop.RunAborted('guard failed')
    values = {name: (lambda: None) for name in inspect.signature(run._promote_anchor).parameters
              if name != 'set_runtime_selection'}
    anchor_build, current_commit = [tmp_path], ['old']
    values.update(args=lambda: SimpleNamespace(worktree=tmp_path, store=tmp_path),
                  publish=lambda: (lambda *args, **kwargs: None),
                  keep_stage=lambda: (lambda *args, **kwargs: nullcontext({})),
                  recipe=lambda: SimpleNamespace(to_dict=lambda: {}),
                  anchor_build=lambda: anchor_build, current_anchor_commit=lambda: current_commit,
                  verify_anchor=lambda: fail, direct_launch=lambda: None)
    with pytest.raises(run.loop.RunAborted, match='guard failed') as exc:
        run._promote_anchor(set_runtime_selection=lambda selected: None, **values)
    assert exc.value.committed_head == 'head'


def test_source_floor_skips_when_no_refresh_is_pending(monkeypatch):
    import inspect

    def fail(*args, **kwargs):
        raise AssertionError('floor lookup should not run')

    monkeypatch.setattr(run, '_select_source_floor', fail)
    values = {name: (lambda: None) for name in inspect.signature(run._ensure_source_floor).parameters
              if name not in ('anchor_recipe', 'a_build', 'apply_source_floor')}
    values['source_floor_refresh'] = lambda: [False]
    assert run._ensure_source_floor(None, None, apply_source_floor=fail, **values) is None


def test_keep_grade_compare_refuses_a_stale_anchor(monkeypatch):
    import inspect
    from types import SimpleNamespace

    monkeypatch.setattr(run, '_cpu_arm', lambda *args: SimpleNamespace(execution_digest='current'))
    values = {name: (lambda: None) for name in inspect.signature(run._keep_grade_runtime_compare).parameters
              if name != 'pair'}
    values['anchor_build'] = lambda: ['build']
    pair = SimpleNamespace(anchor=SimpleNamespace(execution_digest='old'))
    with pytest.raises(run.loop.TailRefused, match='different anchor'):
        run._keep_grade_runtime_compare(pair, **values)


def test_cpu_compare_reads_floor_after_refresh(monkeypatch):
    import inspect
    from types import SimpleNamespace

    floor = [None]
    captured = {}
    monkeypatch.setattr(run, '_cpu_arm', lambda launch, build: build)
    monkeypatch.setattr(run, '_serving_comparison', lambda fn, *args, **kwargs: fn())
    monkeypatch.setattr(run.serving, 'compare', lambda *args, **kwargs: captured.update(kwargs) or 'ok')
    values = {name: (lambda: None) for name in inspect.signature(run._cpu_compare).parameters
              if name not in ('a_build', 'c_build', 'rebind_feedback')}
    values.update(direct_launch=lambda: SimpleNamespace(port=1),
                  feedback_anchor=lambda: [None],
                  ensure_source_floor=lambda: (lambda *args: floor.__setitem__(0, 3.0)),
                  floor=lambda: floor[0], args=lambda: SimpleNamespace(serving_pairs=1),
                  source_instrument=lambda: False)
    assert run._cpu_compare('anchor', 'candidate', **values) == 'ok'
    assert captured['floor_pct'] == 3.0


def test_runtime_owner_propagates_declaration_failure(monkeypatch):
    import inspect
    from . import runtime_calibration

    def fail(**kwargs):
        raise ValueError('invalid declaration')

    monkeypatch.setattr(runtime_calibration, 'declare_statistics', fail)
    values = {name: (lambda: None) for name in inspect.signature(run._install_runtime_owner).parameters
              if name != 'set_runtime_selection'}
    with pytest.raises(ValueError, match='invalid declaration'):
        run._install_runtime_owner(set_runtime_selection=lambda selected: None, **values)


def test_held_claim_capture_records_attempt_and_failure(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace
    from . import measurement_capture

    def fail(*args, **kwargs):
        raise OSError('store unavailable')

    monkeypatch.setattr(measurement_capture, 'ArtifactStore', fail)
    state = {'attempted': False, 'error': None}
    values = {name: (lambda: None) for name in inspect.signature(run._publish_held_claims).parameters
              if name not in ('set_attempted', 'set_evidence', 'set_error')}
    values.update(scheduler_selection=lambda: object(), held_claim_attempted=lambda: state['attempted'],
                  stage_client=lambda: None,
                  args=lambda: SimpleNamespace(cpu_held_intervals_v2='off', out=tmp_path))
    run._publish_held_claims(set_attempted=lambda value: state.__setitem__('attempted', value),
                             set_evidence=lambda value: None,
                             set_error=lambda value: state.__setitem__('error', value), **values)
    assert state == {'attempted': True, 'error': 'OSError: store unavailable'}


def test_preclaim_failure_does_not_publish_after_acquisition(monkeypatch):
    def fail(*args):
        raise AssertionError('post-acquisition must not be marked preclaim')
    monkeypatch.setattr(run, '_publish_preclaim_failure', fail)
    assert not run._publish_preclaim_failure_once(None, object(), None, [object()], [False], 'error')


def test_source_floor_applies_reading_before_clearing_refresh(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace

    reading = SimpleNamespace(floor_pct=1.0, path=tmp_path / 'floor')
    monkeypatch.setattr(run, '_select_source_floor', lambda *args, **kwargs: (None, reading, 'carry'))
    values = {name: (lambda: None) for name in inspect.signature(run._ensure_source_floor).parameters
              if name not in ('anchor_recipe', 'a_build', 'apply_source_floor')}
    refresh, preparation, seen = [True], {}, []
    args = SimpleNamespace(store=tmp_path, serving_instrument='matched', serving_pairs=2,
                           floor_carry_forward='off')
    values.update(source_floor_refresh=lambda: refresh, runtime_preparation=lambda: preparation,
                  current_anchor_commit=lambda: ['head'], args=lambda: args)
    def apply(value, carry):
        seen.append((value, carry, refresh[0]))
    run._ensure_source_floor('recipe', 'build', apply_source_floor=apply, **values)
    assert seen == [(reading, 'carry', True)]
    assert refresh == [False]
    assert preparation['source_comparison_floor'] == str(reading.path)


def test_gate_factory_uses_established_runtime_eligibility(monkeypatch):
    import inspect
    from types import SimpleNamespace
    from . import runtime_combinations

    seen = []
    monkeypatch.setattr(runtime_combinations, 'established', lambda dimension, allowed: seen.append((dimension, allowed)) or False)
    values = {name: None for name in inspect.signature(run._gate_for).parameters if name != 'worker'}
    values.update(recorded_gate=lambda gate: gate, screen_state=None,
                  runtime_enabled=lambda: True, runtime_probe_enabled=False,
                  direct_launch=lambda: object(), cpu_launch=lambda: None,
                  runtime_env_keys={'KNOWN'})
    pair = SimpleNamespace(dimension=object())
    passed, verdicts = run._gate_for(object(), **values)(SimpleNamespace(runtime_pair=pair), [])
    assert not passed and verdicts[0].gate == 'runtime_treatment'
    assert seen == [(pair.dimension, {'KNOWN'})]


def test_cpu_compare_retains_measuring_step_evidence(monkeypatch):
    import inspect
    from types import SimpleNamespace

    arm = SimpleNamespace(execution_digest='digest')
    result, seen = object(), []
    monkeypatch.setattr(run, '_cpu_arm', lambda *args: arm)
    monkeypatch.setattr(run, '_serving_comparison',
                        lambda fn, *args, on_complete, **kwargs: on_complete(result))
    values = {name: (lambda: None) for name in inspect.signature(run._cpu_compare).parameters
              if name not in ('a_build', 'c_build', 'rebind_feedback')}
    values.update(feedback_anchor=lambda: [None], ensure_source_floor=lambda: (lambda *args: None),
                  serving_recipe=lambda: SimpleNamespace(model='model'),
                  retain_measured_source_comparison=lambda: (lambda *args: seen.append(args) or result))
    assert run._cpu_compare('anchor', 'candidate', **values) is result
    assert seen == [(result, 'anchor', 'digest', 'model')]


def test_parser_accepts_cpu_arm_seconds_limit():
    parser = run_args.build_parser(['--planner-only'], description=run.__doc__, run_module=run)
    args = parser.parse_args(['--planner-only', '--worktree', '/tmp/work', '--store', '/tmp/store',
                              '--max-run-cpu-s', '12.5'])
    assert args.max_run_cpu_s == 12.5


def test_run_cpu_claim_hook_charges_claimed_interval(monkeypatch, tmp_path):
    from contextlib import ExitStack, contextmanager
    from ..controller import run_budget

    now = [0.0]
    budget = run_budget.RunBudget(run_budget.Ceilings(cpu_s=4), tmp_path,
                                  clock=lambda: now[0], monotonic=lambda: now[0])
    @contextmanager
    def fake_claim(*args, **kwargs):
        yield {'device_id': 'cpu'}
    monkeypatch.setattr(run.claim, 'hold_cpu', fake_claim)
    token = [None]
    with ExitStack() as stack:
        run._hold_cpu_claim(None, stack, {}, '0', budget=budget, cpu_arm=True, cpu_arm_token=token)
        now[0] += 4
        assert budget.tripped
    assert token == [None]
    assert budget.snapshot()['arm_seconds_claimed']['cpu'] == 4
