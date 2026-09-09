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
    """AST/source guard: the helper is wired into the actual pooled run, not orphaned."""
    source = (Path(__file__).resolve().parent / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    names = {node.attr for node in ast.walk(main) if isinstance(node, ast.Attribute)}
    assert {"WorkerStatusPublisher", "start", "close", "close_failed"} <= names
    assert "publish = status_publisher.publish" in source

    protected = source.split("    try:\n        publish(\"starting\")", 1)[1]
    body, failure = protected.split("    except BaseException as exc:", 1)
    for operation in ("claim.hold()", "reprofile()", "run_pooled()",
                      'status.write_json(args.out, "loop-run.json"'):
        assert operation in body
    assert "status_publisher.close_failed(" in failure
    assert source.index('status.write_json(args.out, "loop-run.json"') < source.index(
        'status_publisher.close("complete"')
    assert "exc, list(latest), hotspot_rows=list(hotspot_rows)" in failure


def test_accumulator_projection_is_validity_gated_and_recovery_refuses_startup():
    source = (Path(__file__).resolve().parent / "run.py").read_text(encoding="utf-8")
    projection = source.split("    def accumulator_state(", 1)[1].split(
        "\n    def actor_health", 1)[0]
    assert '"measurement_validity": validity' in projection
    assert '"historical_compounded_bench_pct": historical_comp' in projection
    assert "if measurement_current else None" in projection
    recovery = source.split("except accumulate.BundleRecoveryRequired as exc:", 1)[1]
    assert "champion.StartupRefused" in recovery
    assert "seed_bundle" in recovery
    assert "not a repair for a corrupt journal" in recovery
    assert "genuinely new explicit baseline" in recovery
    assert "no champion-of-record was inferred" in recovery
