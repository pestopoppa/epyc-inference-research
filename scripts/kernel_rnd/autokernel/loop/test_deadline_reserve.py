"""A run near its granted deadline stops dispatching and ends resumably."""
import pytest

from . import deadline_reserve as dr, provider_breaker as pb


def _rows(values):
    return [{dr.tail_timing.FIELD: {"tail_sessions": [{"held_s": v}]}} for v in values]


def test_trips_only_inside_the_reserve():
    now = [900.0]
    guard = dr.DeadlineReserve(1000.0, 60.0, clock=lambda: now[0])
    assert not guard.tripped
    now[0] = 940.0
    assert guard.tripped


def test_latches_after_tripping():
    now = [950.0]
    guard = dr.DeadlineReserve(1000.0, 60.0, clock=lambda: now[0])
    assert guard.tripped
    now[0] = 0.0
    assert guard.tripped


def test_no_deadline_never_trips():
    assert not dr.DeadlineReserve.from_env(None, env={}).tripped


def test_explicit_reserve_wins_and_is_validated():
    assert dr.resolve_reserve_s(None, env={dr.ENV_RESERVE: "120"}) == 120.0
    with pytest.raises(ValueError):
        dr.resolve_reserve_s(None, env={dr.ENV_RESERVE: "-1"})


def test_default_reserve_is_p95_of_held_tail():
    assert dr.p95_tail_held(_rows(range(1, 101))) == 95.0


def test_too_few_samples_gives_no_default():
    assert dr.p95_tail_held(_rows([500.0] * (dr.MIN_SAMPLES - 1))) is None


def test_unreadable_store_falls_back_to_zero(tmp_path):
    assert dr.resolve_reserve_s(tmp_path / "missing", env={}) == 0.0


def test_terminal_is_resumable_and_operator_stop_wins():
    guard = dr.DeadlineReserve(1.0, clock=lambda: 2.0)
    assert guard.tripped
    terminal = pb.terminal(stop_asked=False, breaker=None, deadline=guard)
    assert terminal == pb.DEADLINE_PARKED
    assert terminal in pb.STOP_TERMINALS and terminal != "stopped"
    assert pb.terminal(stop_asked=True, breaker=None, deadline=guard) == "stopped"
