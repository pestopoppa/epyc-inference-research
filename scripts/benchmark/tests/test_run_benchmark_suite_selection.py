from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_benchmark  # noqa: E402


class _Registry:
    def get_role_config(self, role: str) -> dict:
        return {"candidate_roles": [role]}


class _Suite:
    questions = [object()]


def test_explicit_suite_overrides_role_default_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_benchmark,
        "get_suites_for_role",
        lambda role, registry: ["general", "thinking"],
    )
    monkeypatch.setattr(
        run_benchmark,
        "load_suite",
        lambda name: _Suite() if name == "longcot_mini" else None,
    )

    assert run_benchmark.select_suite_names_for_role(
        "frontdoor", _Registry(), suite_filter="longcot_mini"
    ) == ["longcot_mini"]


def test_unknown_explicit_suite_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_benchmark, "load_suite", lambda name: None)

    with pytest.raises(ValueError, match="unknown benchmark suite"):
        run_benchmark.select_suite_names_for_role(
            "frontdoor", _Registry(), suite_filter="not_a_suite"
        )


def test_all_suites_includes_adapter_suites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_benchmark,
        "get_all_suite_names",
        lambda *, include_adapters=False: ["general", "longcot_mini"]
        if include_adapters
        else ["general"],
    )
    monkeypatch.setattr(run_benchmark, "load_suite", lambda name: _Suite())

    assert run_benchmark.select_suite_names_for_role(
        "frontdoor", _Registry(), all_suites=True, include_vision=False
    ) == ["general", "longcot_mini"]
