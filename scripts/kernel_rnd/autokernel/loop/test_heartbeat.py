"""Hermetic acceptance tests for the worker status/heartbeat lifecycle."""
from __future__ import annotations

import ast
import math
from pathlib import Path
import threading
import time

import pytest

from autokernel.loop.heartbeat import HeartbeatStopError, WorkerStatusPublisher


def _publisher(writer, payload=lambda: {}, *, interval=3600.0, timeout=1.0,
               errors=None):
    errors = [] if errors is None else errors
    return WorkerStatusPublisher(
        writer, payload, interval_s=interval, join_timeout_s=timeout,
        error_sink=errors.append), errors


def _ast_name(node):
    """Return a dotted name for an AST name, attribute, or call target."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _ast_name(node.func)
    return None


def _calls(node):
    if isinstance(node, (list, tuple)):
        return [item for statement in node for item in _calls(statement)]
    return [item for item in ast.walk(node) if isinstance(item, ast.Call)]


def _has_raise_to(node, target):
    return any(isinstance(item, ast.Raise) and item.exc is not None
               and _ast_name(item.exc) == target for item in ast.walk(node))


def test_blocked_inflight_heartbeat_finishes_before_terminal_publication():
    entered = threading.Event()
    release = threading.Event()
    writes = []

    def writer(state, *, step=None):
        writes.append((state, step))
        if state == "starting" and sum(s == "starting" for s, _ in writes) >= 2:
            entered.set()
            assert release.wait(1.0)

    publisher, _ = _publisher(writer, interval=0.01, timeout=1.0)
    publisher.publish("starting", step="claim")
    publisher.start()
    assert entered.wait(1.0)

    closed = threading.Event()

    def close():
        publisher.close("complete")
        closed.set()

    closer = threading.Thread(target=close)
    closer.start()
    time.sleep(0.03)
    assert not closed.is_set()
    assert all(state != "complete" for state, _ in writes)
    release.set()
    closer.join(1.0)

    assert closed.is_set()
    assert writes[-1] == ("complete", "claim")
    time.sleep(0.03)
    assert writes[-1] == ("complete", "claim")


def test_bounded_stop_refuses_terminal_while_heartbeat_is_still_blocked():
    entered = threading.Event()
    release = threading.Event()
    states = []

    def writer(state, *, step=None):
        states.append(state)
        if state == "starting" and states.count("starting") >= 2:
            entered.set()
            release.wait()

    publisher, _ = _publisher(writer, interval=0.01, timeout=0.03)
    publisher.publish("starting")
    publisher.start()
    assert entered.wait(1.0)
    result = []
    finished = threading.Event()

    def close():
        try:
            publisher.close("failed")
        except BaseException as exc:
            result.append(exc)
        finally:
            finished.set()

    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    bounded = finished.wait(0.25)
    # Cleanup happens only after observing the bound; a regression fails instead
    # of leaving an immortal non-daemon test thread behind.
    release.set()
    closer.join(1.0)
    assert bounded, "close waited indefinitely on the in-flight writer"
    assert len(result) == 1
    assert isinstance(result[0], HeartbeatStopError)
    assert "terminal 'failed' not published" in str(result[0])
    assert "failed" not in states

    # A retry is safe after the formerly in-flight heartbeat has observed stop.
    deadline = time.monotonic() + 1.0
    while publisher._thread is not None and publisher._thread.is_alive():
        assert time.monotonic() < deadline
        time.sleep(0.005)
    assert publisher.close("failed") is True
    assert states[-1] == "failed"


def test_close_deadline_includes_write_lock_acquisition():
    entered = threading.Event()
    release = threading.Event()
    states = []

    def writer(state, *, step=None):
        states.append(state)
        if state == "running":
            entered.set()
            release.wait()

    publisher, _ = _publisher(writer, timeout=0.03)
    in_flight = threading.Thread(target=lambda: publisher.publish("running"))
    in_flight.start()
    assert entered.wait(1.0)
    started = time.monotonic()
    result = []
    finished = threading.Event()

    def close():
        try:
            publisher.close("complete")
        except BaseException as exc:
            result.append(exc)
        finally:
            finished.set()

    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    bounded = finished.wait(0.25)
    elapsed = time.monotonic() - started
    release.set()
    in_flight.join(1.0)
    closer.join(1.0)
    assert bounded
    assert elapsed < 0.25
    assert len(result) == 1
    assert isinstance(result[0], HeartbeatStopError)
    assert "status write lock" in str(result[0])
    assert "complete" not in states
    assert publisher.close("complete") is True
    assert states[-1] == "complete"


def test_close_deadline_includes_close_lock_acquisition():
    entered = threading.Event()
    release = threading.Event()
    states = []

    def writer(state, *, step=None):
        states.append(state)
        if state == "complete":
            entered.set()
            release.wait()

    publisher, _ = _publisher(writer, timeout=0.03)
    first = threading.Thread(target=lambda: publisher.close("complete"), daemon=True)
    first.start()
    assert entered.wait(1.0)
    started = time.monotonic()
    result = []
    finished = threading.Event()

    def close_again():
        try:
            publisher.close("failed")
        except BaseException as exc:
            result.append(exc)
        finally:
            finished.set()

    second = threading.Thread(target=close_again, daemon=True)
    second.start()
    bounded = finished.wait(0.25)
    elapsed = time.monotonic() - started
    release.set()
    first.join(1.0)
    second.join(1.0)
    assert bounded
    assert elapsed < 0.25
    assert len(result) == 1
    assert isinstance(result[0], HeartbeatStopError)
    assert "close lock" in str(result[0])
    assert "failed" not in states
    assert states == ["complete"]


@pytest.mark.parametrize("field,value", [
    ("interval", math.nan), ("interval", math.inf),
    ("timeout", math.nan), ("timeout", math.inf),
])
def test_nonfinite_timing_configuration_is_refused(field, value):
    kwargs = {"interval": 1.0, "timeout": 1.0, field: value}
    with pytest.raises(ValueError, match="positive finite"):
        _publisher(lambda state, *, step=None: None, **kwargs)


def test_heartbeat_republishes_starting_until_the_run_advances_to_running():
    states = []
    starting_beat = threading.Event()
    running_beat = threading.Event()

    def writer(state, *, step=None):
        states.append(state)
        if states.count("starting") >= 2:
            starting_beat.set()
        if states.count("running") >= 2:
            running_beat.set()

    publisher, _ = _publisher(writer, interval=0.01)
    publisher.publish("starting")
    publisher.start()
    assert starting_beat.wait(1.0)
    assert "running" not in states
    publisher.publish("running")
    assert running_beat.wait(1.0)
    publisher.close("complete")
    assert states[-1] == "complete"


def test_starting_write_failure_is_followed_by_failed_terminal_attempt():
    states = []

    def writer(state, *, step=None):
        states.append(state)
        if state == "starting":
            raise OSError("starting status unavailable")

    publisher, _ = _publisher(writer)
    with pytest.raises(OSError, match="starting status unavailable"):
        try:
            publisher.publish("starting")
        except BaseException as exc:
            publisher.close_failed(exc)
            raise
    assert states == ["starting", "failed"]


@pytest.mark.parametrize("phase", ["claim", "reprofile", "run"])
def test_body_exception_publishes_failed_for_each_current_run_phase(phase):
    states = []
    publisher, _ = _publisher(
        lambda state, *, step=None: states.append((state, step)))
    publisher.publish("starting")
    publisher.start()

    original = RuntimeError(f"{phase} failed")
    with pytest.raises(RuntimeError) as caught:
        try:
            raise original
        except BaseException as exc:
            publisher.close_failed(exc, step=phase)
            raise
    assert caught.value is original
    assert states[-1] == ("failed", phase)


def test_terminal_writer_error_is_reported_without_replacing_original():
    errors = []

    def writer(state, *, step=None):
        if state == "failed":
            raise OSError("status disk failed")

    publisher, _ = _publisher(writer, errors=errors)
    original = LookupError("run body failed")
    with pytest.raises(LookupError) as caught:
        try:
            raise original
        except BaseException as exc:
            publisher.close_failed(exc)
            raise

    assert caught.value is original
    assert "status disk failed" in errors[0]
    assert any("status disk failed" in note for note in original.__notes__)
    assert publisher.close("failed") is False


def test_close_is_idempotent_and_fences_later_running_writes():
    states = []
    publisher, _ = _publisher(
        lambda state, *, step=None: states.append((state, step)))
    publisher.publish("running", step="measuring")
    assert publisher.close("complete") is True
    assert publisher.close("complete") is False
    assert publisher.publish("running", step="too late") is False
    assert states == [("running", "measuring"), ("complete", "measuring")]


def test_progress_writes_and_last_step_updates_are_serialized():
    entered = threading.Event()
    release = threading.Event()
    writes = []

    def writer(state, *, step=None):
        writes.append((state, step))
        if step == "first":
            entered.set()
            assert release.wait(1.0)

    publisher, _ = _publisher(writer)
    first = threading.Thread(
        target=lambda: publisher.publish("running", step="first"))
    second = threading.Thread(
        target=lambda: publisher.publish("running", step="second"))
    first.start()
    assert entered.wait(1.0)
    second.start()
    time.sleep(0.03)
    assert writes == [("running", "first")]
    release.set()
    first.join(1.0)
    second.join(1.0)
    publisher.close("complete")
    assert writes == [
        ("running", "first"),
        ("running", "second"),
        ("complete", "second"),
    ]


def test_real_main_consumes_lifecycle_across_failures_and_artifact_write():
    """The actual main path brackets work and artifact writing with terminal states."""
    source = (Path(__file__).resolve().parent / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_assignments = [node for node in main.body if isinstance(node, ast.Assign)]
    publisher_assignment = next(node for node in main_assignments
                                if any(isinstance(target, ast.Name)
                                       and target.id == "status_publisher"
                                       for target in node.targets))
    publish_binding = next(node for node in main_assignments
                           if any(isinstance(target, ast.Name) and target.id == "publish"
                                  for target in node.targets))
    assert isinstance(publisher_assignment.value, ast.Call)
    assert _ast_name(publisher_assignment.value.func) == "heartbeat.WorkerStatusPublisher"
    assert _ast_name(publish_binding.value) == "status_publisher.publish"

    lifecycle = [node for node in ast.walk(main) if isinstance(node, ast.Try)
                 and any(_ast_name(handler.type) == "BaseException"
                         for handler in node.handlers if handler.type)
                 and any(_ast_name(call.func) == "status_publisher.close_failed"
                         for handler in node.handlers for call in _calls(handler))]
    assert len(lifecycle) == 1
    lifecycle = lifecycle[0]
    assert publisher_assignment.lineno < publish_binding.lineno < lifecycle.lineno

    body_calls = _calls(lifecycle.body)
    starting = [call for call in body_calls
                if _ast_name(call.func) == "publish"
                and call.args and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "starting"]
    starts = [call for call in body_calls
              if _ast_name(call.func) == "status_publisher.start"]
    claims = [call for call in body_calls if _ast_name(call.func) == "claim.hold"]
    profiling = [call for call in body_calls if _ast_name(call.func) == "reprofile"]
    pooled_runs = [call for call in body_calls if _ast_name(call.func) == "run_pooled"]
    artifact_writes = [call for call in body_calls
                       if _ast_name(call.func) == "status.write_json"
                       and len(call.args) >= 2
                       and isinstance(call.args[1], ast.Constant)
                       and call.args[1].value == "loop-run.json"]
    assert len(starting) == len(starts) == len(pooled_runs) == len(artifact_writes) == 1
    assert claims and profiling
    assert starting[0].lineno < starts[0].lineno < pooled_runs[0].lineno
    assert starts[0].lineno < claims[0].lineno < profiling[0].lineno
    assert pooled_runs[0].lineno < artifact_writes[0].lineno

    failures = [handler for handler in lifecycle.handlers
                if _ast_name(handler.type) == "BaseException"]
    assert len(failures) == 1
    failed_closes = [call for call in _calls(failures[0])
                     if _ast_name(call.func) == "status_publisher.close_failed"]
    assert len(failed_closes) == 1
    assert any(isinstance(arg, ast.Name) and arg.id == failures[0].name
               for arg in failed_closes[0].args)
    assert any(isinstance(item, ast.Raise) and item.exc is None
               for item in ast.walk(failures[0]))

    success_closes = [call for call in _calls(lifecycle.orelse)
                      if _ast_name(call.func) == "status_publisher.close"
                      and call.args and isinstance(call.args[0], ast.Constant)
                      and call.args[0].value == "complete"]
    assert len(success_closes) == 1
    assert artifact_writes[0].lineno < success_closes[0].lineno


def test_accumulator_projection_is_validity_gated_and_recovery_refuses_startup():
    source = (Path(__file__).resolve().parent / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    accumulator = next(node for node in ast.walk(main)
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "accumulator_state")
    projection = next(node.value for node in ast.walk(accumulator)
                      if isinstance(node, ast.Return)
                      and isinstance(node.value, ast.Dict))
    projected = {key.value: value for key, value in zip(projection.keys, projection.values)
                 if isinstance(key, ast.Constant) and isinstance(key.value, str)}

    assert isinstance(projected["measurement_validity"], ast.Name)
    assert projected["measurement_validity"].id == "validity"
    current_gain = projected["compounded_bench_pct"]
    historical_gain = projected["historical_compounded_bench_pct"]
    assert isinstance(current_gain, ast.Name) and current_gain.id == "comp"
    assert isinstance(historical_gain, ast.Name) and historical_gain.id == "historical_comp"
    assignments = {target.id: node.value for node in ast.walk(accumulator)
                   if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    validity_gate = assignments["measurement_current"]
    assert isinstance(validity_gate, ast.Compare)
    assert isinstance(validity_gate.left, ast.Name) and validity_gate.left.id == "validity"
    assert _ast_name(validity_gate.comparators[0]) == "accumulate.MEASUREMENT_CURRENT"
    assert isinstance(validity_gate.ops[0], ast.Eq)
    current_gain = assignments[current_gain.id]
    historical_gain = assignments[historical_gain.id]
    assert isinstance(current_gain, ast.IfExp)
    assert _ast_name(current_gain.test) == "measurement_current"
    assert isinstance(current_gain.orelse, ast.Constant) and current_gain.orelse.value is None
    assert isinstance(historical_gain, ast.IfExp)
    assert _ast_name(historical_gain.test) == "measurement_current"
    assert isinstance(historical_gain.body, ast.Constant) and historical_gain.body.value is None
    assert isinstance(historical_gain.orelse, ast.Call)
    assert _ast_name(historical_gain.orelse.func) == "round"

    recovery = [node for node in ast.walk(main) if isinstance(node, ast.Try)
                and any(_ast_name(call.func) == "accumulate.load_bundle"
                        for call in _calls(node))
                and any(_ast_name(handler.type) == "accumulate.BundleRecoveryRequired"
                        for handler in node.handlers if handler.type)]
    assert len(recovery) == 1
    handler = next(handler for handler in recovery[0].handlers
                   if _ast_name(handler.type) == "accumulate.BundleRecoveryRequired")
    explicit_epoch = next(node for node in handler.body if isinstance(node, ast.If)
                          and _ast_name(node.test) == "args.new_anchor_epoch")
    assert any(_ast_name(call.func) == "new_epoch.start_new_anchor_epoch"
               for call in _calls(explicit_epoch.body))
    assert _has_raise_to(explicit_epoch, "champion.StartupRefused")
    assert _has_raise_to(ast.Module(body=explicit_epoch.orelse, type_ignores=[]),
                         "champion.StartupRefused")
