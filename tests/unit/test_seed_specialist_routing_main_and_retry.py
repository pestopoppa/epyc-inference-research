"""Coverage-focused tests for specialist routing main/retry control flow."""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_BENCH))


_ROUTING_FILES = (
    "seed_specialist_routing.py",
    "seed_specialist_routing_v2.py",
)

_RETRIEVAL_OVERRIDE_KEYS = (
    "cost_lambda",
    "confidence_threshold",
    "confidence_estimator",
    "confidence_trim_ratio",
    "confidence_min_neighbors",
    "warm_probability_hit",
    "warm_probability_miss",
    "warm_cost_fallback_s",
    "cold_cost_fallback_s",
    "calibrated_confidence_threshold",
    "conformal_margin",
    "risk_control_enabled",
    "risk_budget_id",
    "risk_gate_min_samples",
    "risk_abstain_target_role",
    "risk_gate_rollout_ratio",
    "risk_gate_kill_switch",
    "risk_budget_guardrail_min_events",
    "risk_budget_guardrail_max_abstain_rate",
    "prior_strength",
)


def _load_module(name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(name, _BENCH / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bind_research_pool(mod, monkeypatch, questions):
    """EVL-12 A2: --question-ids must read the research pool via the path-bound loader.

    A stale same-named module is planted in sys.modules; if the bare import comes
    back, it raises.
    """
    stale = ModuleType("question_pool")
    stale.load_questions_by_ids = Mock(side_effect=AssertionError("bare-import copy used"))
    monkeypatch.setitem(sys.modules, "question_pool", stale)
    pool_mod = ModuleType("question_pool")
    pool_mod.load_pool = Mock(return_value={"suite_a": list(questions)})
    monkeypatch.setattr(
        mod._seeding_sampling, "_load_research_benchmark_module", Mock(return_value=pool_mod),
    )
    return pool_mod


def _install_orchestration_module_tree_stubs(monkeypatch) -> None:
    orchestration_pkg = ModuleType("orchestration")
    orchestration_pkg.__path__ = []  # type: ignore[attr-defined]
    repl_pkg = ModuleType("orchestration.repl_memory")
    repl_pkg.__path__ = []  # type: ignore[attr-defined]
    replay_pkg = ModuleType("orchestration.repl_memory.replay")
    replay_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "orchestration", orchestration_pkg)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory", repl_pkg)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay", replay_pkg)
    orchestration_pkg.repl_memory = repl_pkg
    repl_pkg.replay = replay_pkg


def _install_sqlite_connect_tracker(monkeypatch) -> list[sqlite3.Connection]:
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def _connect(*args, **kwargs):  # noqa: ANN002, ANN003
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _connect)
    return opened


def _close_sqlite_connections(conns: list[sqlite3.Connection]) -> None:
    for conn in conns:
        try:
            conn.close()
        except Exception:
            pass


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        suites=["thinking"],
        roles=["frontdoor"],
        modes=["direct"],
        profile="infra-stable",
        question_ids=None,
        sample_size=1,
        seed=7,
        url="http://localhost:8000",
        timeout=10,
        dry_run=False,
        output=None,
        skip_cache=False,
        cooldown=0.0,
        no_dedup=False,
        continuous=False,
        continuous_interval=1,
        resume=None,
        preflight=False,
        stats=False,
        no_escalation_chains=False,
        three_way=False,
        tui=False,
        rebuild_pool=False,
        no_pool=False,
        debug=False,
        debug_batch_size=2,
        debug_threshold=0.2,
        debug_auto_commit=False,
        debug_dry_run=False,
        debug_replay=False,
        cost_lambda=None,
        confidence_threshold=None,
        confidence_estimator=None,
        confidence_trim_ratio=None,
        confidence_min_neighbors=None,
        warm_probability_hit=None,
        warm_probability_miss=None,
        warm_cost_fallback_s=None,
        cold_cost_fallback_s=None,
        calibrated_confidence_threshold=None,
        conformal_margin=None,
        risk_control_enabled=False,
        risk_budget_id=None,
        risk_gate_min_samples=None,
        risk_abstain_target_role=None,
        risk_gate_rollout_ratio=None,
        risk_gate_kill_switch=False,
        risk_budget_guardrail_min_events=None,
        risk_budget_guardrail_max_abstain_rate=None,
        prior_strength=None,
        evolve=False,
    )


def _retrieval_config_stub() -> SimpleNamespace:
    return SimpleNamespace(**{k: None for k in _RETRIEVAL_OVERRIDE_KEYS})


def _role_result(*, passed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        role="frontdoor",
        mode="direct",
        passed=passed,
        answer="answer",
        error=None,
        error_type=None,
        tokens_generated=1,
        tool_output_tokens=0,
        elapsed_seconds=0.01,
        difficulty_score=0.0,
        difficulty_band="",
        factual_risk_score=0.0,
        factual_risk_band="",
        role_history=[],
        delegation_events=[],
        delegation_diagnostics={},
        tools_used=[],
        tools_called=[],
        tap_offset_bytes=0,
        tap_length_bytes=0,
        repl_tap_offset_bytes=0,
        repl_tap_length_bytes=0,
        cost_dimensions={},
        think_harder_attempted=False,
        think_harder_succeeded=False,
        cheap_first_attempted=False,
        cheap_first_passed=False,
        grammar_enforced=False,
        parallel_tools_used=False,
        cache_affinity_bonus=0.0,
        skills_retrieved=[],
        skill_ids=["type_skill_1"],
        budget_diagnostics={},
        tool_results_cleared=False,
        compaction_triggered=False,
        compaction_tokens_saved=0,
        tool_chains=[],
        think_harder_expected_roi=0.0,
        compression_metrics={},
    )


def _three_way_result(mod, *, metadata: dict | None = None, reward: float = 1.0):
    return mod.ThreeWayResult(
        suite="suite_a",
        question_id="q1",
        prompt="p",
        expected="e",
        role_results={"SELF:direct": SimpleNamespace(passed=reward >= 0.5)},
        rewards={mod.ACTION_SELF_DIRECT: reward},
        metadata=metadata or {},
        rewards_injected=1,
    )


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_stats_mode_calls_print_stats(file_name):
    mod = _load_module(f"routing_main_stats_{file_name}", file_name)
    args = _base_args()
    args.stats = True
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "print_stats") as mock_print_stats,
    ):
        mod.main()
    mock_print_stats.assert_called_once()


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_rebuild_pool_mode_prints_summary(file_name, monkeypatch, capsys):
    mod = _load_module(f"routing_main_pool_{file_name}", file_name)
    args = _base_args()
    args.rebuild_pool = True

    # EVL-12 A2: a same-named module on sys.path must NOT capture --rebuild-pool.
    stale = ModuleType("question_pool")
    stale.build_pool = Mock(side_effect=AssertionError("bare-import copy used"))
    monkeypatch.setitem(sys.modules, "question_pool", stale)
    pool_mod = ModuleType("question_pool")
    pool_mod.build_pool = Mock(return_value={"suite_a": 3, "suite_b": 1})
    loader = Mock(return_value=pool_mod)
    monkeypatch.setattr(mod._seeding_sampling, "_load_research_benchmark_module", loader)

    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        mod.main()

    out = capsys.readouterr().out
    assert "Pool rebuilt" in out
    assert "suite_a" in out
    loader.assert_called_with("question_pool")
    pool_mod.build_pool.assert_called_once_with()
    stale.build_pool.assert_not_called()


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_one_shot_calls_runner(file_name):
    mod = _load_module(f"routing_main_3way_{file_name}", file_name)
    args = _base_args()
    args.three_way = True

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "run_batch_3way", return_value=[]) as mock_batch,
        patch.object(mod, "print_3way_summary") as mock_summary,
    ):
        mod.main()

    assert mock_batch.call_count == 1
    mock_summary.assert_called_once_with([])


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_preflight_failure_exits(file_name):
    mod = _load_module(f"routing_main_preflight_{file_name}", file_name)
    args = _base_args()
    args.preflight = True

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "run_preflight", return_value=False),
        pytest.raises(SystemExit) as excinfo,
    ):
        mod.main()

    assert excinfo.value.code == 1


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_resume_propagates_session_id(file_name):
    mod = _load_module(f"routing_main_resume_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.resume = "resume_session_123"

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "run_batch_3way", return_value=[]) as mock_batch,
        patch.object(mod, "print_3way_summary"),
    ):
        mod.main()

    assert mock_batch.call_args.kwargs["session_id"] == "resume_session_123"
    assert mod.state.session_id == "resume_session_123"


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_debug_requires_three_way_warns(file_name, tmp_path):
    mod = _load_module(f"routing_main_debug_warn_{file_name}", file_name)
    args = _base_args()
    args.debug = True
    args.output = str(tmp_path / f"{Path(file_name).stem}_debug_warn.json")

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_build_role_mode_combos", return_value=[("frontdoor", "direct")]),
        patch.object(mod, "run_batch", return_value=[]),
        patch.object(mod.logger, "warning") as warning_mock,
    ):
        mod.main()

    joined = " ".join(str(c.args[0]) for c in warning_mock.call_args_list if c.args)
    assert "--debug requires --3way" in joined


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_continuous_single_batch_then_stop(file_name):
    mod = _load_module(f"routing_main_3way_cont_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.continuous = True

    def _run_once(**_kwargs):
        mod.state.shutdown = True
        return []

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "run_batch_3way", side_effect=_run_once) as batch_mock,
        patch.object(mod, "print_3way_summary") as summary_mock,
        patch.object(mod.time, "sleep"),
    ):
        mod.state.shutdown = False
        mod.main()

    assert batch_mock.call_count == 1
    summary_mock.assert_called_once_with([])


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_legacy_continuous_single_batch_then_stop(file_name):
    mod = _load_module(f"routing_main_legacy_cont_{file_name}", file_name)
    args = _base_args()
    args.continuous = True

    def _run_once(**_kwargs):
        mod.state.shutdown = True
        return [object()]

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "run_batch", side_effect=_run_once) as batch_mock,
        patch.object(mod, "print_batch_summary") as summary_mock,
        patch.object(mod.time, "sleep"),
    ):
        mod.state.shutdown = False
        mod.main()

    assert batch_mock.call_count == 1
    assert summary_mock.call_count == 1


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_debug_auto_commit_enables_debug(file_name, tmp_path):
    mod = _load_module(f"routing_main_debug_auto_{file_name}", file_name)
    args = _base_args()
    args.debug_auto_commit = True
    args.output = str(tmp_path / f"{Path(file_name).stem}_debug_auto.json")

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_build_role_mode_combos", return_value=[("frontdoor", "direct")]),
        patch.object(mod, "run_batch", return_value=[]),
    ):
        mod.main()

    assert args.debug is True


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_tui_progress_updates(file_name, monkeypatch):
    mod = _load_module(f"routing_main_tui_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.tui = True

    class _FakeTUI:
        def __init__(self, session_id: str):
            self.session_id = session_id
            self.updates = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update_progress(self, *_args, **_kwargs):
            self.updates += 1

    tui_holder: dict[str, _FakeTUI] = {}

    def _mk_tui(session_id: str):
        tui = _FakeTUI(session_id)
        tui_holder["tui"] = tui
        return tui

    tui_mod = ModuleType("seeding_tui")
    tui_mod.SeedingTUI = _mk_tui
    monkeypatch.setitem(sys.modules, "seeding_tui", tui_mod)

    def _run_once(**kwargs):
        kwargs["on_progress"](1, 1, "suite_a", "q1", "p")
        return []

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "run_batch_3way", side_effect=_run_once),
        patch.object(mod, "print_3way_summary"),
    ):
        mod.main()

    assert tui_holder["tui"].updates == 1


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_continuous_health_recovery_branches(file_name):
    mod = _load_module(f"routing_main_3way_health_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.continuous = True

    # Pass 1: health down -> recovery success.
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", side_effect=[False, True]),
        patch.object(mod, "_attempt_recovery", return_value=True),
        patch.object(mod, "run_batch_3way", side_effect=lambda **_k: (setattr(mod.state, "shutdown", True), [])[1]),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.time, "sleep"),
    ):
        mod.state.shutdown = False
        mod.main()

    # Pass 2: health down -> recovery failed -> backoff sleep.
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", return_value=False),
        patch.object(mod, "_attempt_recovery", return_value=False),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.time, "sleep", side_effect=lambda *_a, **_k: setattr(mod.state, "shutdown", True)),
    ):
        mod.state.shutdown = False
        mod.main()


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_continuous_batch_exception_path(file_name):
    mod = _load_module(f"routing_main_3way_batch_exc_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.continuous = True

    def _raise_once(**_kwargs):
        mod.state.shutdown = True
        raise RuntimeError("boom")

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "run_batch_3way", side_effect=_raise_once),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.time, "sleep"),
        patch.object(mod.logger, "error") as err_mock,
    ):
        mod.state.shutdown = False
        mod.main()

    assert any("Batch failed" in str(c.args[0]) for c in err_mock.call_args_list if c.args)


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_continuous_periodic_hooks(file_name, monkeypatch):
    mod = _load_module(f"routing_main_periodic_hooks_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.continuous = True
    args.debug_replay = True
    args.evolve = True

    _install_orchestration_module_tree_stubs(monkeypatch)
    tracked_conns = _install_sqlite_connect_tracker(monkeypatch)

    claude_mod = ModuleType("src.pipeline_monitor.claude_debugger")
    claude_mod.ClaudeDebugger = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "src.pipeline_monitor.claude_debugger", claude_mod)

    traj_mod = ModuleType("orchestration.repl_memory.replay.trajectory")
    traj_mod.TrajectoryExtractor = lambda _reader: SimpleNamespace(extract_complete=lambda **_k: [object()])
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.trajectory", traj_mod)

    replay_mod = ModuleType("orchestration.repl_memory.replay.engine")
    replay_metrics = SimpleNamespace(num_trajectories=1, routing_accuracy=1.0, avg_reward=1.0)
    replay_mod.ReplayEngine = lambda: SimpleNamespace(run_with_metrics=lambda *_a, **_k: replay_metrics)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.engine", replay_mod)

    scorer_mod = ModuleType("orchestration.repl_memory.q_scorer")
    scorer_mod.ScoringConfig = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.q_scorer", scorer_mod)

    reader_mod = ModuleType("orchestration.repl_memory.progress_logger")
    reader_mod.ProgressReader = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.progress_logger", reader_mod)

    evo_mod = ModuleType("orchestration.repl_memory.skill_evolution")
    evo_mod.OutcomeTracker = lambda: object()
    evo_mod.EvolutionMonitor = lambda _sb: SimpleNamespace(
        run_evolution_cycle=lambda **_k: SimpleNamespace(
            skills_evaluated=1,
            skills_promoted=1,
            skills_decayed=0,
            skills_deprecated=0,
        )
    )
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_evolution", evo_mod)

    bank_mod = ModuleType("orchestration.repl_memory.skill_bank")
    bank_mod.SkillBank = lambda **_k: SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_bank", bank_mod)

    counter = {"n": 0}

    def _run_batch(**_kwargs):
        counter["n"] += 1
        if counter["n"] >= 10:
            mod.state.shutdown = True
        return [object()]

    orig_exists = mod.Path.exists

    def _exists(path_obj):
        if str(path_obj).endswith("orchestration/repl_memory/sessions/skills.db"):
            return True
        return orig_exists(path_obj)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod.Path, "exists", _exists),
        patch.object(mod, "run_batch_3way", side_effect=_run_batch),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.time, "sleep"),
    ):
        mod.state.shutdown = False
        mod.main()
    _close_sqlite_connections(tracked_conns)

    assert counter["n"] == 10


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_continuous_periodic_hooks_no_trajectories_and_exceptions(file_name, monkeypatch):
    mod = _load_module(f"routing_main_periodic_hooks_edge_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.continuous = True
    args.debug_replay = True
    args.evolve = True

    _install_orchestration_module_tree_stubs(monkeypatch)
    tracked_conns = _install_sqlite_connect_tracker(monkeypatch)

    evo_mod = ModuleType("orchestration.repl_memory.skill_evolution")
    evo_mod.OutcomeTracker = lambda: object()
    evo_mod.EvolutionMonitor = lambda _sb: SimpleNamespace(
        run_evolution_cycle=lambda **_k: (_ for _ in ()).throw(RuntimeError("evolve boom"))
    )
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_evolution", evo_mod)

    bank_mod = ModuleType("orchestration.repl_memory.skill_bank")
    bank_mod.SkillBank = lambda **_k: SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_bank", bank_mod)

    scorer_mod = ModuleType("orchestration.repl_memory.q_scorer")
    scorer_mod.ScoringConfig = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.q_scorer", scorer_mod)

    reader_mod = ModuleType("orchestration.repl_memory.progress_logger")
    reader_mod.ProgressReader = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.progress_logger", reader_mod)

    replay_mod = ModuleType("orchestration.repl_memory.replay.engine")
    replay_mod.ReplayEngine = lambda: SimpleNamespace(run_with_metrics=lambda *_a, **_k: SimpleNamespace(num_trajectories=1, routing_accuracy=1.0, avg_reward=1.0))
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.engine", replay_mod)

    # Pass 1: no trajectories path.
    traj_mod = ModuleType("orchestration.repl_memory.replay.trajectory")
    traj_mod.TrajectoryExtractor = lambda _reader: SimpleNamespace(extract_complete=lambda **_k: [])
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.trajectory", traj_mod)

    counter = {"n": 0}

    def _run_batch(**_kwargs):
        counter["n"] += 1
        if counter["n"] >= 10:
            mod.state.shutdown = True
        return [object()]

    orig_exists = mod.Path.exists

    def _exists(path_obj):
        if str(path_obj).endswith("orchestration/repl_memory/sessions/skills.db"):
            return True
        return orig_exists(path_obj)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod.Path, "exists", _exists),
        patch.object(mod, "run_batch_3way", side_effect=_run_batch),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.time, "sleep"),
        patch.object(mod.logger, "warning") as warn_mock,
    ):
        mod.state.shutdown = False
        mod.main()
    _close_sqlite_connections(tracked_conns)

    # Pass 2: replay extractor exception path.
    counter["n"] = 0
    traj_mod.TrajectoryExtractor = lambda _reader: SimpleNamespace(
        extract_complete=lambda **_k: (_ for _ in ()).throw(RuntimeError("replay boom"))
    )
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod.Path, "exists", _exists),
        patch.object(mod, "run_batch_3way", side_effect=_run_batch),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.time, "sleep"),
        patch.object(mod.logger, "warning") as warn2_mock,
    ):
        mod.state.shutdown = False
        mod.main()
    _close_sqlite_connections(tracked_conns)

    joined = " ".join(str(c.args[0]) for c in (warn_mock.call_args_list + warn2_mock.call_args_list) if c.args)
    assert "Periodic replay failed" in joined
    assert "Periodic evolution failed" in joined


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_debug_replay_skill_metrics_and_evolve_report(file_name, monkeypatch):
    mod = _load_module(f"routing_main_debug_skill_metrics_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.debug_replay = True
    args.evolve = True

    _install_orchestration_module_tree_stubs(monkeypatch)

    traj_mod = ModuleType("orchestration.repl_memory.replay.trajectory")
    traj_mod.TrajectoryExtractor = lambda _reader: SimpleNamespace(extract_complete=lambda **_k: [object()])
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.trajectory", traj_mod)

    replay_mod = ModuleType("orchestration.repl_memory.replay.engine")
    base_metrics = SimpleNamespace(
        num_trajectories=1,
        routing_accuracy=1.0,
        routing_accuracy_by_type={"thinking": 1.0},
        cumulative_reward=1.0,
        avg_reward=1.0,
        q_convergence_step=1,
        replay_duration_seconds=0.1,
        escalation_precision=1.0,
        escalation_recall=1.0,
        ece_global=0.0,
        brier_global=0.0,
        conformal_coverage=1.0,
        conformal_risk=0.0,
    )
    replay_mod.ReplayEngine = lambda: SimpleNamespace(run_with_metrics=lambda *_a, **_k: base_metrics)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.engine", replay_mod)

    skill_replay_mod = ModuleType("orchestration.repl_memory.replay.skill_replay")
    skill_replay_mod.SkillBankConfig = lambda: object()
    skill_replay_mod.SkillAwareReplayEngine = lambda **_k: SimpleNamespace(
        run_with_skill_metrics=lambda *_a, **_k2: SimpleNamespace(
            base_metrics=base_metrics,
            total_skills_retrieved=3,
            skill_coverage=0.9,
            avg_skills_per_step=1.2,
        )
    )
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.skill_replay", skill_replay_mod)

    scorer_mod = ModuleType("orchestration.repl_memory.q_scorer")
    scorer_mod.ScoringConfig = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.q_scorer", scorer_mod)

    reader_mod = ModuleType("orchestration.repl_memory.progress_logger")
    reader_mod.ProgressReader = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.progress_logger", reader_mod)

    evo_mod = ModuleType("orchestration.repl_memory.skill_evolution")
    evo_mod.OutcomeTracker = lambda: object()
    evo_mod.EvolutionMonitor = lambda _sb: SimpleNamespace(
        run_evolution_cycle=lambda **_k: SimpleNamespace(
            skills_evaluated=1,
            skills_promoted=1,
            skills_decayed=0,
            skills_deprecated=0,
            redistillation_candidates=["skill_a"],
        )
    )
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_evolution", evo_mod)

    bank_mod = ModuleType("orchestration.repl_memory.skill_bank")
    bank_mod.SkillBank = lambda **_k: SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_bank", bank_mod)

    orig_exists = mod.Path.exists

    def _exists(path_obj):
        if str(path_obj).endswith("orchestration/repl_memory/sessions/skills.db"):
            return True
        return orig_exists(path_obj)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod.Path, "exists", _exists),
        patch.object(mod, "run_batch_3way", return_value=[]),
        patch.object(mod, "print_3way_summary"),
    ):
        mod.main()


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_debug_replay_importerror_fallback(file_name, monkeypatch):
    mod = _load_module(f"routing_main_debug_import_fallback_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.debug_replay = True

    _install_orchestration_module_tree_stubs(monkeypatch)

    traj_mod = ModuleType("orchestration.repl_memory.replay.trajectory")
    traj_mod.TrajectoryExtractor = lambda _reader: SimpleNamespace(extract_complete=lambda **_k: [object()])
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.trajectory", traj_mod)

    replay_mod = ModuleType("orchestration.repl_memory.replay.engine")
    replay_mod.ReplayEngine = lambda: SimpleNamespace(
        run_with_metrics=lambda *_a, **_k: SimpleNamespace(
            num_trajectories=1,
            routing_accuracy=1.0,
            routing_accuracy_by_type={"thinking": 1.0},
            cumulative_reward=1.0,
            avg_reward=1.0,
            q_convergence_step=1,
            replay_duration_seconds=0.1,
            escalation_precision=1.0,
            escalation_recall=1.0,
            ece_global=0.0,
            brier_global=0.0,
            conformal_coverage=1.0,
            conformal_risk=0.0,
        )
    )
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.engine", replay_mod)

    scorer_mod = ModuleType("orchestration.repl_memory.q_scorer")
    scorer_mod.ScoringConfig = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.q_scorer", scorer_mod)

    reader_mod = ModuleType("orchestration.repl_memory.progress_logger")
    reader_mod.ProgressReader = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.progress_logger", reader_mod)

    orig_import = builtins.__import__

    def _import(name, *args_, **kwargs):
        if name == "orchestration.repl_memory.replay.skill_replay":
            raise ImportError("forced")
        return orig_import(name, *args_, **kwargs)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch("builtins.__import__", side_effect=_import),
        patch.object(mod, "run_batch_3way", return_value=[]),
        patch.object(mod, "print_3way_summary"),
    ):
        mod.main()


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_evolve_no_skills_db_logs_info(file_name, monkeypatch):
    mod = _load_module(f"routing_main_evolve_no_db_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.evolve = True

    _install_orchestration_module_tree_stubs(monkeypatch)

    evo_mod = ModuleType("orchestration.repl_memory.skill_evolution")
    evo_mod.OutcomeTracker = lambda: object()
    evo_mod.EvolutionMonitor = lambda _sb: SimpleNamespace(run_evolution_cycle=lambda **_k: None)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_evolution", evo_mod)

    bank_mod = ModuleType("orchestration.repl_memory.skill_bank")
    bank_mod.SkillBank = lambda **_k: SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_bank", bank_mod)

    orig_exists = mod.Path.exists

    def _exists(path_obj):
        if str(path_obj).endswith("orchestration/repl_memory/sessions/skills.db"):
            return False
        return orig_exists(path_obj)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod.Path, "exists", _exists),
        patch.object(mod, "run_batch_3way", return_value=[]),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.logger, "info") as info_mock,
    ):
        mod.main()
    assert any("No skills.db found" in str(c.args[0]) for c in info_mock.call_args_list if c.args)


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_debug_replay_path(file_name, monkeypatch):
    mod = _load_module(f"routing_main_debug_replay_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.debug = True
    args.debug_replay = True

    _install_orchestration_module_tree_stubs(monkeypatch)

    claude_mod = ModuleType("src.pipeline_monitor.claude_debugger")
    claude_mod.ClaudeDebugger = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "src.pipeline_monitor.claude_debugger", claude_mod)

    traj_mod = ModuleType("orchestration.repl_memory.replay.trajectory")
    traj_mod.TrajectoryExtractor = lambda _reader: SimpleNamespace(extract_complete=lambda **_k: [object()])
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.trajectory", traj_mod)

    engine_mod = ModuleType("orchestration.repl_memory.replay.engine")
    metrics = SimpleNamespace(
        num_trajectories=1,
        routing_accuracy=1.0,
        routing_accuracy_by_type={"thinking": 1.0},
        cumulative_reward=1.0,
        avg_reward=1.0,
        q_convergence_step=1,
        replay_duration_seconds=0.01,
        escalation_precision=1.0,
        escalation_recall=1.0,
        ece_global=0.0,
        brier_global=0.0,
        conformal_coverage=1.0,
        conformal_risk=0.0,
    )
    engine_mod.ReplayEngine = lambda: SimpleNamespace(run_with_metrics=lambda *_a, **_k: metrics)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.engine", engine_mod)

    scorer_mod = ModuleType("orchestration.repl_memory.q_scorer")
    scorer_mod.ScoringConfig = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.q_scorer", scorer_mod)

    reader_mod = ModuleType("orchestration.repl_memory.progress_logger")
    reader_mod.ProgressReader = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.progress_logger", reader_mod)

    skill_replay_mod = ModuleType("orchestration.repl_memory.replay.skill_replay")
    skill_replay_mod.SkillAwareReplayEngine = lambda **_k: SimpleNamespace(
        run_with_skill_metrics=lambda *_a, **_k2: SimpleNamespace(
            base_metrics=metrics,
            total_skills_retrieved=1,
            skill_coverage=1.0,
            avg_skills_per_step=1.0,
        )
    )
    skill_replay_mod.SkillBankConfig = lambda: object()
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.replay.skill_replay", skill_replay_mod)

    skill_bank_mod = ModuleType("orchestration.repl_memory.skill_bank")
    skill_bank_mod.SkillBank = lambda **_k: SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_bank", skill_bank_mod)

    orig_exists = mod.Path.exists

    def _exists(path_obj):
        if str(path_obj).endswith("orchestration/repl_memory/sessions/skills.db"):
            return False
        return orig_exists(path_obj)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod.Path, "exists", _exists),
        patch.object(mod, "run_batch_3way", return_value=[]),
        patch.object(mod, "print_3way_summary"),
    ):
        mod.main()


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_evolve_init_failure_logs_error(file_name, monkeypatch):
    mod = _load_module(f"routing_main_evolve_fail_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.evolve = True

    evo_mod = ModuleType("orchestration.repl_memory.skill_evolution")

    class _OutcomeTracker:
        def __init__(self):
            raise RuntimeError("boom")

    evo_mod.OutcomeTracker = _OutcomeTracker
    monkeypatch.setitem(sys.modules, "orchestration.repl_memory.skill_evolution", evo_mod)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "run_batch_3way", return_value=[]),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.logger, "error") as err_mock,
    ):
        mod.main()

    joined = " ".join(str(c.args[0]) for c in err_mock.call_args_list if c.args)
    assert "[SKILLBANK] OutcomeTracker init failed" in joined
    assert "[EVOLVE] --evolve passed but OutcomeTracker is None!" in joined


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_continuous_unrecoverable_break(file_name):
    mod = _load_module(f"routing_main_3way_unrecoverable_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.continuous = True

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "MAX_RECOVERY_ATTEMPTS", 1),
        patch.object(mod, "_check_server_health", side_effect=[False, False]),
        patch.object(mod, "_attempt_recovery", return_value=False),
        patch.object(mod.time, "sleep"),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.logger, "error") as err_mock,
    ):
        mod.state.shutdown = False
        mod.main()
    assert any("unrecoverable" in str(c.args[0]) for c in err_mock.call_args_list if c.args)


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_three_way_continuous_no_results_wait_loop(file_name):
    mod = _load_module(f"routing_main_3way_no_results_wait_{file_name}", file_name)
    args = _base_args()
    args.three_way = True
    args.continuous = True

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "run_batch_3way", side_effect=lambda **_k: []),
        patch.object(mod, "print_3way_summary"),
        patch.object(mod.time, "sleep", side_effect=lambda *_a, **_k: setattr(mod.state, "shutdown", True)) as sleep_mock,
    ):
        mod.state.shutdown = False
        mod.main()
    assert sleep_mock.call_count >= 1


def test_main_three_way_continuous_question_ids_override_break(tmp_path, monkeypatch):
    mod = _load_module("routing_main_3way_qids_break", "seed_specialist_routing.py")
    args = _base_args()
    args.three_way = True
    args.continuous = True
    args.question_ids = str(tmp_path / "qids.json")
    Path(args.question_ids).write_text(json.dumps(["suite_a/q1"]))
    questions = [{"id": "q1", "suite": "suite_a", "prompt": "p", "expected": "e"}]

    _bind_research_pool(mod, monkeypatch, questions)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "run_batch_3way", return_value=[object()]) as run_mock,
        patch.object(mod, "print_3way_summary"),
    ):
        mod.state.shutdown = False
        mod.main()
    assert run_mock.call_count == 1


def test_main_question_ids_all_question_ids_dict_format(tmp_path, monkeypatch):
    mod = _load_module("routing_main_qids_all_question_ids", "seed_specialist_routing.py")
    args = _base_args()
    args.three_way = True
    args.question_ids = str(tmp_path / "qids_payload.json")
    Path(args.question_ids).write_text(json.dumps({"all_question_ids": ["suite_a/q1"]}))
    questions = [{"id": "q1", "suite": "suite_a", "prompt": "p", "expected": "e"}]

    _bind_research_pool(mod, monkeypatch, questions)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "run_batch_3way", return_value=[]) as batch_mock,
        patch.object(mod, "print_3way_summary"),
    ):
        mod.main()

    assert batch_mock.call_args.kwargs["questions_override"] == questions


def test_main_question_ids_missing_file_exits(tmp_path):
    mod = _load_module("routing_main_qids_missing", "seed_specialist_routing.py")
    args = _base_args()
    args.three_way = True
    args.question_ids = str(tmp_path / "missing.json")

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        pytest.raises(SystemExit) as excinfo,
    ):
        mod.main()

    assert excinfo.value.code == 1


def test_main_question_ids_forces_dry_run_and_passes_override(tmp_path, monkeypatch):
    mod = _load_module("routing_main_qids_success", "seed_specialist_routing.py")
    args = _base_args()
    args.three_way = True
    args.question_ids = str(tmp_path / "qids.json")
    args.dry_run = False

    qid_file = Path(args.question_ids)
    qid_file.write_text(json.dumps(["suite_a/q1"]))
    questions = [{"id": "q1", "suite": "suite_a", "prompt": "p", "expected": "e"}]

    _bind_research_pool(mod, monkeypatch, questions)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "run_batch_3way", return_value=[]) as mock_batch,
        patch.object(mod, "print_3way_summary"),
    ):
        mod.main()

    called = mock_batch.call_args.kwargs
    assert called["questions_override"] == questions
    assert called["dry_run"] is True


def test_main_question_ids_invalid_json_shape_exits(tmp_path):
    mod = _load_module("routing_main_qids_bad_shape", "seed_specialist_routing.py")
    args = _base_args()
    args.three_way = True
    args.question_ids = str(tmp_path / "qids_bad.json")
    Path(args.question_ids).write_text(json.dumps({"unexpected": ["q1"]}))

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        pytest.raises(SystemExit) as excinfo,
    ):
        mod.main()

    assert excinfo.value.code == 1


def test_main_question_ids_empty_lookup_exits(tmp_path, monkeypatch):
    mod = _load_module("routing_main_qids_empty_lookup", "seed_specialist_routing.py")
    args = _base_args()
    args.three_way = True
    args.question_ids = str(tmp_path / "qids.json")
    Path(args.question_ids).write_text(json.dumps(["suite_a/q1"]))

    pool_mod = _bind_research_pool(mod, monkeypatch, questions=[])

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        pytest.raises(SystemExit) as excinfo,
    ):
        mod.main()

    assert excinfo.value.code == 1
    pool_mod.load_pool.assert_called_once_with(None, warn_stale=False)


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_legacy_one_shot_writes_json(file_name, tmp_path):
    mod = _load_module(f"routing_main_legacy_{file_name}", file_name)
    args = _base_args()
    args.output = str(tmp_path / f"{Path(file_name).stem}_out.json")

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_build_role_mode_combos", return_value=[("frontdoor", "direct")]),
        patch.object(mod, "run_batch", return_value=[]),
    ):
        mod.main()

    assert Path(args.output).exists()
    payload = json.loads(Path(args.output).read_text())
    assert payload["config"]["suites"] == args.suites
    assert payload["results"] == []


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_legacy_one_shot_results_and_auto_output(file_name, tmp_path):
    mod = _load_module(f"routing_main_legacy_auto_out_{file_name}", file_name)
    args = _base_args()
    args.output = None

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_build_role_mode_combos", return_value=[("frontdoor", "direct")]),
        patch.object(mod, "run_batch", return_value=[object()]),
        patch.object(mod, "print_batch_summary") as print_summary_mock,
        patch("dataclasses.asdict", return_value={"ok": True}),
        patch.object(mod.Path, "mkdir"),
        patch("builtins.open", mock_open()),
    ):
        mod.main()

    assert print_summary_mock.call_count == 1


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_legacy_continuous_health_and_batch_error_paths(file_name):
    mod = _load_module(f"routing_main_legacy_health_{file_name}", file_name)
    args = _base_args()
    args.continuous = True

    # Health-down branch with failed recovery and sleep loop.
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_check_server_health", return_value=False),
        patch.object(mod, "_attempt_recovery", return_value=False),
        patch.object(mod.time, "sleep", side_effect=lambda *_a, **_k: setattr(mod.state, "shutdown", True)),
    ):
        mod.state.shutdown = False
        mod.main()

    # Batch raises HealthCheckError branch.
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "run_batch", side_effect=lambda **_k: (setattr(mod.state, "shutdown", True), (_ for _ in ()).throw(mod.HealthCheckError("dead")))[1]),
        patch.object(mod.time, "sleep"),
        patch.object(mod.logger, "warning") as warn_mock,
    ):
        mod.state.shutdown = False
        mod.main()
    assert any("API died during batch" in str(c.args[0]) for c in warn_mock.call_args_list if c.args)


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_legacy_continuous_unrecoverable_and_recovery_success(file_name):
    mod = _load_module(f"routing_main_legacy_unrecoverable_{file_name}", file_name)
    args = _base_args()
    args.continuous = True

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "MAX_RECOVERY_ATTEMPTS", 1),
        patch.object(mod, "_check_server_health", side_effect=[False, False]),
        patch.object(mod, "_attempt_recovery", return_value=False),
        patch.object(mod.time, "sleep"),
        patch.object(mod.logger, "error") as err_mock,
    ):
        mod.state.shutdown = False
        mod.main()
    assert any("unrecoverable" in str(c.args[0]) for c in err_mock.call_args_list if c.args)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_check_server_health", side_effect=[False, True]),
        patch.object(mod, "_attempt_recovery", return_value=True),
        patch.object(mod, "run_batch", side_effect=lambda **_k: (setattr(mod.state, "shutdown", True), [object()])[1]),
        patch.object(mod, "print_batch_summary"),
        patch.object(mod.time, "sleep"),
    ):
        mod.state.shutdown = False
        mod.main()


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_main_legacy_continuous_sleep_interval(file_name):
    mod = _load_module(f"routing_main_legacy_sleep_interval_{file_name}", file_name)
    args = _base_args()
    args.continuous = True
    args.continuous_interval = 2

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch.object(mod, "_build_retrieval_config_from_args", return_value=_retrieval_config_stub()),
        patch.object(mod, "_deduplicate_roles", return_value=(args.roles, {"frontdoor": "frontdoor"})),
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "run_batch", side_effect=lambda **_k: [object()]),
        patch.object(mod, "print_batch_summary"),
        patch.object(mod.time, "sleep", side_effect=lambda *_a, **_k: setattr(mod.state, "shutdown", True)) as sleep_mock,
    ):
        mod.state.shutdown = False
        mod.main()
    assert sleep_mock.call_count >= 1


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_print_3way_summary_hurt_and_neutral_paths(file_name, capsys):
    mod = _load_module(f"routing_summary_hurt_neutral_{file_name}", file_name)
    results = [
        _three_way_result(mod, metadata={"tools_hurt": True}, reward=0.0),
        _three_way_result(mod, metadata={}, reward=1.0),
    ]
    mod.print_3way_summary(results)
    out = capsys.readouterr().out
    assert "Tools hurt: 1" in out
    assert "Tools neutral: 1" in out


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_run_batch_3way_recovery_progress_and_outcomes(file_name):
    mod = _load_module(f"routing_batch_recovery_{file_name}", file_name)

    question = {"id": "q1", "suite": "suite_a", "prompt": "p1", "expected": "e1"}
    role_results = {"SELF:direct": _role_result(passed=False)}
    rewards = {mod.ACTION_SELF_DIRECT: 0.0}
    metadata = {"all_infra": True}
    progress = Mock()
    outcomes = Mock()
    client = MagicMock()

    kwargs = dict(
        suites=["suite_a"],
        sample_per_suite=1,
        seed=1,
        url="http://localhost:8000",
        timeout=30,
        session_id="sess",
        dry_run=False,
        on_progress=progress,
        outcome_tracker=outcomes,
    )
    if "questions_override" in mod.run_batch_3way.__code__.co_varnames:
        kwargs["questions_override"] = [question]

    with (
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "load_seen_questions", return_value=set()),
        patch.object(mod, "sample_unseen_questions", return_value=[question]),
        patch.object(mod, "evaluate_question_3way", return_value=(role_results, rewards, metadata)),
        patch.object(mod, "_inject_3way_rewards_http", return_value={"acknowledged": 1, "submitted": 1, "failed": 0}),
        patch.object(mod, "_attempt_recovery", return_value=False) as recovery_mock,
        patch.object(mod, "checkpoint_result"),
        patch.object(mod, "record_seen"),
        patch("httpx.Client", return_value=client),
        patch.object(mod.time, "sleep") as sleep_mock,
    ):
        mod.state.shutdown = False
        result = mod.run_batch_3way(**kwargs)

    assert len(result) == 1
    assert progress.call_count == 1
    assert recovery_mock.call_count == 1
    sleep_mock.assert_any_call(30)
    assert outcomes.record_outcome.call_count == 1


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_run_batch_3way_recovery_success_branch(file_name):
    mod = _load_module(f"routing_batch_recovery_success_{file_name}", file_name)
    question = {"id": "q1", "suite": "suite_a", "prompt": "p1", "expected": "e1"}
    role_results = {"SELF:direct": _role_result(passed=False)}
    rewards = {mod.ACTION_SELF_DIRECT: 0.0}
    metadata = {"all_infra": True}

    kwargs = dict(
        suites=["suite_a"],
        sample_per_suite=1,
        seed=1,
        url="http://localhost:8000",
        timeout=30,
        session_id="sess",
        dry_run=False,
    )
    if "questions_override" in mod.run_batch_3way.__code__.co_varnames:
        kwargs["questions_override"] = [question]

    with (
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "load_seen_questions", return_value=set()),
        patch.object(mod, "sample_unseen_questions", return_value=[question]),
        patch.object(mod, "evaluate_question_3way", return_value=(role_results, rewards, metadata)),
        patch.object(mod, "_inject_3way_rewards_http", return_value={"acknowledged": 1, "submitted": 1, "failed": 0}),
        patch.object(mod, "_attempt_recovery", return_value=True) as recovery_mock,
        patch.object(mod, "checkpoint_result"),
        patch.object(mod, "record_seen"),
        patch("httpx.Client", return_value=MagicMock()),
        patch.object(mod.time, "sleep") as sleep_mock,
    ):
        mod.state.shutdown = False
        out = mod.run_batch_3way(**kwargs)

    assert len(out) == 1
    assert recovery_mock.call_count == 1
    assert sleep_mock.call_count == 0


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_run_batch_3way_shutdown_before_loop_breaks(file_name):
    mod = _load_module(f"routing_batch_shutdown_{file_name}", file_name)
    question = {"id": "q1", "suite": "suite_a", "prompt": "p1", "expected": "e1"}

    with (
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "load_seen_questions", return_value=set()),
        patch.object(mod, "sample_unseen_questions", return_value=[question]),
        patch.object(mod, "evaluate_question_3way") as eval_mock,
        patch("httpx.Client", return_value=MagicMock()),
    ):
        mod.state.shutdown = True
        out = mod.run_batch_3way(
            suites=["suite_a"],
            sample_per_suite=1,
            seed=1,
            url="http://localhost:8000",
            timeout=30,
            session_id="sess",
        )
    assert out == []
    assert eval_mock.call_count == 0


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_run_batch_3way_retry_poll_wait_and_shutdown_break(file_name, monkeypatch):
    mod = _load_module(f"routing_batch_retry_poll_break_{file_name}", file_name)
    initial_question = {"id": "q1", "suite": "suite_a", "prompt": "p1", "expected": "e1"}
    eval_tuple = (
        {"SELF:direct": _role_result(passed=True)},
        {mod.ACTION_SELF_DIRECT: 1.0},
        {},
    )

    class _Debugger:
        batch_count = 3

        def add_diagnostic(self, _diag):  # noqa: ANN001
            pass

        def end_question(self):
            pass

        def pop_retries(self):
            return [("suite_a", "q1")], {"suite_a"}

        def flush(self):
            pass

    diag_mod = ModuleType("src.pipeline_monitor.diagnostic")
    diag_mod.build_diagnostic = lambda **kwargs: dict(kwargs)
    diag_mod.append_diagnostic = Mock()
    if "src" not in sys.modules:
        monkeypatch.setitem(sys.modules, "src", ModuleType("src"))
    if "src.pipeline_monitor" not in sys.modules:
        monkeypatch.setitem(sys.modules, "src.pipeline_monitor", ModuleType("src.pipeline_monitor"))
    monkeypatch.setitem(sys.modules, "src.pipeline_monitor.diagnostic", diag_mod)

    def _eval_once(*_a, **_k):
        mod.state.shutdown = True
        return eval_tuple

    with (
        patch.object(mod, "_check_server_health", side_effect=[True, False, True]),
        patch.object(mod, "load_seen_questions", return_value=set()),
        patch.object(mod, "sample_unseen_questions", return_value=[initial_question]),
        patch.object(mod, "evaluate_question_3way", side_effect=_eval_once),
        patch.object(mod, "_inject_3way_rewards_http", return_value={"acknowledged": 1, "submitted": 1, "failed": 0}),
        patch.object(mod, "checkpoint_result"),
        patch.object(mod, "record_seen"),
        patch("httpx.Client", return_value=MagicMock()),
        patch.object(mod.time, "sleep") as sleep_mock,
    ):
        mod.state.shutdown = False
        out = mod.run_batch_3way(
            suites=["suite_a"],
            sample_per_suite=1,
            seed=1,
            url="http://localhost:8000",
            timeout=30,
            session_id="sess",
            debugger=_Debugger(),
        )
    assert len(out) == 1
    sleep_mock.assert_any_call(1)


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_run_batch_3way_debugger_retry_flow(file_name, monkeypatch):
    mod = _load_module(f"routing_retry_{file_name}", file_name)

    initial_question = {"id": "q1", "suite": "suite_a", "prompt": "p1", "expected": "e1"}
    generalize_question = {"id": "q2", "suite": "suite_a", "prompt": "p2", "expected": "e2"}
    eval_tuple = (
        {"SELF:direct": _role_result(passed=True)},
        {mod.ACTION_SELF_DIRECT: 1.0},
        {},
    )

    class _Debugger:
        batch_count = 9

        def __init__(self) -> None:
            self.ended = 0
            self.added = 0
            self.flushed = 0

        def add_diagnostic(self, _diag) -> None:
            self.added += 1

        def end_question(self) -> None:
            self.ended += 1

        def pop_retries(self):
            return [("suite_a", "q1")], {"suite_a"}

        def flush(self) -> None:
            self.flushed += 1

    debugger = _Debugger()

    diag_mod = ModuleType("src.pipeline_monitor.diagnostic")
    diag_mod.build_diagnostic = lambda **kwargs: dict(kwargs)
    diag_mod.append_diagnostic = Mock()
    if "src" not in sys.modules:
        monkeypatch.setitem(sys.modules, "src", ModuleType("src"))
    if "src.pipeline_monitor" not in sys.modules:
        monkeypatch.setitem(sys.modules, "src.pipeline_monitor", ModuleType("src.pipeline_monitor"))
    monkeypatch.setitem(sys.modules, "src.pipeline_monitor.diagnostic", diag_mod)

    client = MagicMock()

    def _sample(_suites, sample_per_suite, _seen, *_, **_kwargs):
        if sample_per_suite == 1:
            return [initial_question]
        return [generalize_question]

    run_kwargs = dict(
        suites=["suite_a"],
        sample_per_suite=1,
        seed=1,
        url="http://localhost:8000",
        timeout=30,
        session_id="sess",
        dry_run=False,
        cooldown=0.0,
        debugger=debugger,
    )
    if "questions_override" in mod.run_batch_3way.__code__.co_varnames:
        run_kwargs["questions_override"] = [initial_question]

    with (
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "load_seen_questions", return_value=set()),
        patch.object(mod, "sample_unseen_questions", side_effect=_sample),
        patch.object(mod, "evaluate_question_3way", side_effect=[eval_tuple, eval_tuple, eval_tuple, eval_tuple]),
        patch.object(mod, "_inject_3way_rewards_http", return_value={"acknowledged": 1, "submitted": 1, "failed": 0}),
        patch.object(mod, "checkpoint_result") as checkpoint_mock,
        patch.object(mod, "record_seen"),
        patch("httpx.Client", return_value=client),
        patch.object(mod.time, "sleep"),
    ):
        mod.state.shutdown = False
        results = mod.run_batch_3way(**run_kwargs)

    assert len(results) == 4
    assert checkpoint_mock.call_count == 4
    assert debugger.added >= 4
    assert debugger.ended >= 2
    assert debugger.flushed == 1


@pytest.mark.parametrize("file_name", _ROUTING_FILES)
def test_run_batch_3way_debugger_exception_is_non_fatal(file_name, monkeypatch):
    mod = _load_module(f"routing_retry_nonfatal_{file_name}", file_name)

    question = {"id": "q1", "suite": "suite_a", "prompt": "p1", "expected": "e1"}
    eval_tuple = (
        {"SELF:direct": _role_result(passed=True)},
        {mod.ACTION_SELF_DIRECT: 1.0},
        {},
    )

    class _Debugger:
        def add_diagnostic(self, _diag) -> None:  # pragma: no cover - not reached
            pass

        def end_question(self) -> None:  # pragma: no cover - not reached
            pass

        def pop_retries(self):
            return [], set()

        def flush(self) -> None:
            self.flushed = True

    diag_mod = ModuleType("src.pipeline_monitor.diagnostic")

    def _raise_diag(**_kwargs):
        raise RuntimeError("diag boom")

    diag_mod.build_diagnostic = _raise_diag
    diag_mod.append_diagnostic = Mock()
    if "src" not in sys.modules:
        monkeypatch.setitem(sys.modules, "src", ModuleType("src"))
    if "src.pipeline_monitor" not in sys.modules:
        monkeypatch.setitem(sys.modules, "src.pipeline_monitor", ModuleType("src.pipeline_monitor"))
    monkeypatch.setitem(sys.modules, "src.pipeline_monitor.diagnostic", diag_mod)

    kwargs = dict(
        suites=["suite_a"],
        sample_per_suite=1,
        seed=1,
        url="http://localhost:8000",
        timeout=30,
        session_id="sess",
        debugger=_Debugger(),
    )
    if "questions_override" in mod.run_batch_3way.__code__.co_varnames:
        kwargs["questions_override"] = [question]

    with (
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "load_seen_questions", return_value=set()),
        patch.object(mod, "sample_unseen_questions", return_value=[question]),
        patch.object(mod, "evaluate_question_3way", return_value=eval_tuple),
        patch.object(mod, "checkpoint_result"),
        patch.object(mod, "record_seen"),
        patch("httpx.Client", return_value=MagicMock()),
        patch.object(mod.logger, "warning") as warning_mock,
    ):
        mod.state.shutdown = False
        out = mod.run_batch_3way(**kwargs)

    assert len(out) == 1
    joined = " ".join(str(c.args[0]) for c in warning_mock.call_args_list if c.args)
    assert "Debugger error" in joined
